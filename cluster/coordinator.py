"""Two backends with the same shape of contract:
  step(tokens, start_pos) -> argmax per input position (teacher-forced)
  crop(length)            -> keep KV in sync with the accepted context

LocalBackend runs every layer in-process (the single-machine baseline).
RingHead runs its own block, ships activations to the tail, and the tail
returns argmaxes straight back (direct-return, no relay).
"""
import os
import socket
import time

import mlx.core as mx
import numpy as np

from . import protocol as P
from .config import ToyConfig
from .fastmodel import FastModel
from .model import ToyModel, new_kv

DEBUG = __import__("os").environ.get("CLUSTER_DEBUG", "") not in ("", "0")


class StageLost(RuntimeError):
    """A node stopped answering. Retriable by rebuild, not by retrying the frame.

    The distinction matters to the host: `closed` and `timeout` both mean "that
    node's KV is gone and the chain's state is now suspect", so the host must
    rebuild and replay rather than resend.
    """

    def __init__(self, msg, kind="closed"):
        super().__init__(msg)
        self.kind = kind


class LocalBackend:
    def __init__(self, cfg: ToyConfig, model_dir=None):
        self.cfg = cfg
        if model_dir:
            from .realmodel import RealModel
            self.model = RealModel(model_dir, lo=0, hi=cfg.n_layers,
                                   need_embed=True, need_head=True)
        else:
            self.model = ToyModel(cfg)
        self.kv = new_kv(cfg.n_layers)

    def step(self, tokens, start_pos, kind=None):
        h = self.model.embed(tokens)
        t0 = time.perf_counter()
        h = self.model.forward_range(h, start_pos, 0, self.cfg.n_layers, self.kv)
        lg, _ = self.model.logits_all(h)
        mx.eval(lg)
        tau = (time.perf_counter() - t0) * 1e3
        return mx.argmax(lg[0], axis=-1).tolist(), tau, 0.0, 0.0

    def crop(self, length):
        self.model.crop_kv(self.kv, 0, self.cfg.n_layers, length)

    def reset(self):
        """KV 全部丟掉。每個新批次都必須呼叫：batch 軸的長度跟這一輪的 B 綁死，
        上一輪殘留的 KV 是 [B_old, ...]，換一批人就是形狀不符（實測會炸
        concatenate，shapes (1,0,14,64) vs (2,9,14,64)）。"""
        self.kv = new_kv(self.cfg.n_layers)

    def forward_rows(self, rows, start_pos, kind=None, valid_from=None):
        """單機版的 batch：同樣的 [B,L,H] 路徑，用來當「多台 batch」的對照基準。"""
        B, L = len(rows), len(rows[0])
        if valid_from is None:
            valid_from = [0] * B
        vf = mx.array(list(valid_from), mx.int32)
        rope_pos = mx.arange(start_pos, start_pos + L)[None, :] - vf[:, None]
        t0 = time.perf_counter()
        x = self.model.embed_rows(rows)
        h = self.model.forward_range(x, start_pos, 0, self.cfg.n_layers, self.kv,
                                     rope_pos=rope_pos, valid_from=vf)
        lg, _ = self.model.logits_all(h)
        mx.eval(lg)
        tau = (time.perf_counter() - t0) * 1e3
        if lg.shape[0] == 1:
            toks = mx.argmax(lg[0], axis=-1).tolist()[-1:]
        else:
            toks = mx.argmax(lg, axis=-1)[:, -1].tolist()
        return toks, tau, 0.0, 0.0


class RingHead:
    def __init__(self, cfg: ToyConfig, lo: int, hi: int, out_sock, in_sock,
                 timeout_s=15.0, fast=False, ctx_len=640, model_dir=None):
        self.cfg = cfg
        self.lo, self.hi = lo, hi
        self.out_sock, self.in_sock = out_sock, in_sock
        self.timeout_s = timeout_s
        self.out_sock.settimeout(timeout_s)
        self.in_sock.settimeout(timeout_s)
        self.fast = FastModel(cfg, ctx_len) if fast else None
        if model_dir:
            from .realmodel import RealModel
            self.model = RealModel(model_dir, lo=lo, hi=hi,
                                   need_embed=True, need_head=False)
        else:
            self.model = None if fast else ToyModel(cfg)
        self.kv = None if fast else new_kv(cfg.n_layers)
        self.chunk_id = 0
        # 這條鏈有多快，決定了「多久沒回應才算異常」。用固定 15s 等於對 4ms
        # 的往返寬容 3750 倍：真卡死時要多等 15 秒，而遠程鏈路抖一下又不會被
        # 當成異常。所以基線要量，門檻跟著它走。
        self.lat_ms = 0.0          # 每輪延遲（EWMA）
        self.last_rtt_ms = 0.0     # 最近一次心跳往返
        self.slow_rounds = 0
        self.pings = 0

    def _exchange(self, h, kind, meta):
        """送出 [B,L,H]、等一輪結果回來。回傳 (arr, meta, latency_ms)。

        把「送出 → 等 → 逾時判斷 → 心跳探活」抽在這裡，是因為 batch 路徑要走
        完全一樣的錯誤語意：掉線要立刻知道（closed），慢要能區分（timeout/hung）。
        """
        self.chunk_id += 1
        meta = dict(meta)
        meta["chunk_id"] = self.chunk_id
        meta["n_in"] = int(h.shape[1])
        meta["batch"] = int(h.shape[0])
        t1 = time.perf_counter()
        try:
            P.send_frame(self.out_sock, kind, meta, np.array(h))
        except OSError as exc:
            raise StageLost(f"stage link broke while sending: {exc}", "closed") from exc

        deadline_ms = self.deadline_ms()
        self.in_sock.settimeout(deadline_ms / 1e3)
        P.arm(self.in_sock)
        try:
            _, rmeta, arr = P.recv_frame(self.in_sock)
        except socket.timeout as exc:
            waited = (time.perf_counter() - t1) * 1e3
            alive, note = self._probe_alive()
            kind_s = "timeout" if alive else "hung"
            why = ("alive but slow (heartbeat answered in "
                   f"{self.last_rtt_ms:.1f}ms) -- degraded, not dead"
                   if alive else
                   "no RESULT and no heartbeat answer -- open socket, "
                   "machine not computing")
            raise StageLost(
                f"stage {why}: waited {waited:.0f}ms, baseline {self.lat_ms:.1f}ms, "
                f"deadline {deadline_ms:.0f}ms (chunk {self.chunk_id}){note}",
                kind_s) from exc
        except ConnectionError as exc:
            raise StageLost(
                "stage closed the connection (crashed or was killed); "
                "in-flight chunk is lost", "closed") from exc
        latency = (time.perf_counter() - t1) * 1e3
        self.lat_ms = latency if self.lat_ms == 0 else 0.3 * latency + 0.7 * self.lat_ms
        if rmeta.get("chunk_id") != self.chunk_id:
            raise ValueError(
                f"out-of-order result: {rmeta.get('chunk_id')} != {self.chunk_id}")
        return arr, rmeta, latency

    def step(self, tokens, start_pos, kind=P.VERIFY):
        t0 = time.perf_counter()
        if self.fast is not None:
            x = self.fast.embed(tokens)
            if len(tokens) == 1:
                h = self.fast.step(x[0], mx.array(start_pos, mx.int32),
                                   self.lo, self.hi)[None]
            else:
                h = self.fast.forward_batch(x, start_pos, self.lo, self.hi)
        else:
            h = self.model.embed(tokens)
            h = self.model.forward_range(h, start_pos, self.lo, self.hi, self.kv)
        mx.eval(h)
        tau_h = (time.perf_counter() - t0) * 1e3
        if os.environ.get("CLUSTER_TRACE"):
            print(f"[trace head {self.lo}:{self.hi}] n={len(tokens)} "
                  f"pos={start_pos} tau={tau_h:.1f}ms", flush=True)

        arr, rmeta, latency = self._exchange(h, kind, {"start_pos": start_pos})
        if DEBUG:
            print(f"[head] recv arr{None if arr is None else arr.shape} "
                  f"first={arr.tolist()[0]!r}", flush=True)
        return arr.tolist(), tau_h, rmeta.get("tau_ms", 0.0), latency

    def forward_rows(self, rows, start_pos, kind=P.PREFILL, valid_from=None):
        """B 條序列一起走一步（prefill 或 decode）。回傳每條的下一個 token。

        兩個不變量：
        - **物理位置同步**：KV 的 batch 軸共用同一組索引（`start_pos`），所以所有
          列一起前進。等長時 `valid_from` 全 0；**不等長就左補齊**，`valid_from[r]`
          是第 r 列第一個真實 token 的物理位置，前面那些 pad 進得了 KV 但看不見
          （per-row 遮罩），rope 也用它算出「內容位置」。
        - 每列的**內容位置**可以不同（`start_pos + i - valid_from[r]`），這才是
          Qwen 這種絕對位置模型該看到的位置編號。
        """
        if self.model is None:
            raise SystemExit("forward_rows 只支援 realmodel 路徑")
        B, L = len(rows), len(rows[0])
        if valid_from is None:
            valid_from = [0] * B
        vf = mx.array(list(valid_from), mx.int32)
        # 物理位置 → 內容位置：左邊被 pad 吃掉多少，就要往前平移多少
        rope_pos = (mx.arange(start_pos, start_pos + L)[None, :] - vf[:, None])
        t0 = time.perf_counter()
        x = self.model.embed_rows(rows)                       # [B, L, H]
        h = self.model.forward_range(x, start_pos, self.lo, self.hi, self.kv,
                                     rope_pos=rope_pos, valid_from=vf)
        mx.eval(h)
        tau_h = (time.perf_counter() - t0) * 1e3
        arr, rmeta, latency = self._exchange(h, kind, {
            "start_pos": start_pos,
            "valid_from": [int(v) for v in valid_from],
        })
        toks = arr.tolist()
        n = len(rows)
        # B=1 的 prefill 會回「每個位置一個 argmax」（那是投機驗證在用的舊語意），
        # 這裡只要最後一個位置的下一個 token。
        if n == 1 and len(toks) != 1:
            toks = toks[-1:]
        if len(toks) != n:
            raise ValueError(f"batch mismatch: sent {n} rows, got {len(toks)} tokens")
        return toks, tau_h, rmeta.get("tau_ms", 0.0), latency


    def deadline_ms(self):
        """門檻 = 20× 實測基線，夾在 [0.5s, timeout_s]。沒量到基線時用硬上限。"""
        if self.lat_ms <= 0:
            return self.timeout_s * 1e3
        return max(500.0, min(self.timeout_s * 1e3, 20.0 * self.lat_ms))

    def ping(self, budget=None, raise_on_fail=True):
        """問一次「還在嗎」。開工前先問，比跑到一半才發現鏈路死了便宜得多——
        遠程鏈路放半小時後是不是還活著，只有問了才知道。"""
        budget = budget or min(2.0, max(0.25, 4.0 * self.last_rtt_ms / 1e3))
        t0 = time.perf_counter()
        self.chunk_id += 1
        cid = self.chunk_id
        try:
            self.in_sock.settimeout(budget)
            P.send_frame(self.out_sock, P.HEARTBEAT, {"chunk_id": cid}, None)
            P.arm(self.in_sock)
            while True:
                ftype, meta, _ = P.recv_frame(self.in_sock)
                if ftype == P.HEARTBEAT and meta.get("chunk_id") == cid:
                    self.last_rtt_ms = (time.perf_counter() - t0) * 1e3
                    self.pings += 1
                    return self.last_rtt_ms
                # 遲到的 RESULT：他其實活著，只是慢了；不吞掉結論，只記活著
                self.last_rtt_ms = (time.perf_counter() - t0) * 1e3
                self.pings += 1
                return self.last_rtt_ms
        except socket.timeout as exc:
            if not raise_on_fail:
                return None
            raise StageLost(
                f"no heartbeat answer within {budget:.2f}s: the link is open but "
                f"the stage is not responding (baseline round {self.lat_ms:.1f}ms)",
                "timeout") from exc
        except OSError as exc:
            if not raise_on_fail:
                return None
            raise StageLost(f"heartbeat failed: {exc}", "closed") from exc

    def _probe_alive(self, budget=1.0):
        """硬門檻到了之後再問一句：是慢，還是卡死。"""
        self.last_rtt_ms = 0.0          # 一定要清零，否則會拿舊的回應當現在的
        self.ping(budget=budget, raise_on_fail=False)
        return self.last_rtt_ms > 0, ""

    def crop(self, length):
        if self.fast is not None:
            return              # slab keeps stale rows; the mask makes them inert
        self.model.crop_kv(self.kv, self.lo, self.hi, length)

    # -- recovery -------------------------------------------------------
    def rebind(self, out_sock, in_sock):
        """Point at a freshly built chain; KV is dropped because it is suspect."""
        self.out_sock, self.in_sock = out_sock, in_sock
        self.out_sock.settimeout(self.timeout_s)
        self.in_sock.settimeout(self.timeout_s)
        self.lat_ms = 0.0            # 換了機器，舊基線不算數

    def reset(self):
        if self.fast is not None:
            self.fast = FastModel(self.cfg, self.fast.ctx)
        else:
            self.kv = new_kv(self.cfg.n_layers)
        try:
            P.send_frame(self.out_sock, P.RESET, {})
        except OSError as exc:
            raise StageLost(f"cannot reset the chain: {exc}", "closed") from exc

    def replay(self, tokens):
        """Rebuild every stage's KV by re-running the known prefix.

        The head is the only node holding the authoritative token prefix, so
        replay is the one recovery that cannot be silently wrong: after it,
        every stage's KV matches the head's, whatever the failure was.
        """
        if not tokens:
            return
        self.reset()
        self.step(list(tokens), 0, kind=P.PREFILL)


def generate(backend, prompt, max_new_tokens, block=4, mode="greedy", metrics=None,
             recovery=None, drafter=None, stats=None, pacing=None):
    """`recovery(exc, ctx)` is called when a node dies; it must make the chain
    consistent with `ctx` again, or raise. Returning means "retry this step"."""
    ctx = list(prompt)
    if hasattr(backend, "ping"):
        backend.ping()               # 開工前先確認鏈路還在，別跑到一半才發現
    tp = time.perf_counter()
    a, tau_h, tau_t, lat = backend.step(ctx, 0, kind=P.PREFILL)
    prefill_ms = (time.perf_counter() - tp) * 1e3
    if metrics is not None:
        metrics.prefill_ms = prefill_ms
    ctx.append(a[-1])
    if metrics:
        metrics.round(tau_h, tau_t, lat, len(ctx) - 1, 1)

    while len(ctx) - len(prompt) < max_new_tokens:
        try:
            if mode == "greedy":
                pos = len(ctx) - 1
                a, tau_h, tau_t, lat = backend.step([ctx[-1]], pos)
                ctx.append(a[-1])
                n_in, accepted = 1, 1
            else:
                pos = len(ctx) - 1
                # 草稿可以比 block 短（甚至為空）：空的話這一輪就退化成 greedy，
                # 只虧一次判斷，不虧 block。這比硬湊一個爛草稿划算。
                # block 可以由 pacing 線上決定（k 高就多送、k 低就不投機）
                blk = block
                if pacing is not None:
                    blk = pacing.block()
                if blk <= 0:
                    a, tau_h, tau_t, lat = backend.step([ctx[-1]], pos)
                    ctx.append(a[-1])
                    if pacing is not None:
                        pacing.observe(1, 0, tau_h + tau_t, lat)
                    if metrics:
                        metrics.round(tau_h, tau_t, lat, 1, 1)
                    continue
                if drafter is not None:
                    drafts = list(drafter.propose(ctx, blk))[:blk]
                else:
                    drafts = [ctx[-1]] * blk            # 舊的佔位草稿（見 DESIGN.md）
                if stats is not None and hasattr(stats, "note"):
                    stats.note(drafts)
                if not drafts:
                    a, tau_h, tau_t, lat = backend.step([ctx[-1]], pos)
                    ctx.append(a[-1])
                    n_in, accepted = 1, 1
                    if pacing is not None:
                        pacing.observe(1, 0, tau_h + tau_t, lat)
                else:
                    a, tau_h, tau_t, lat = backend.step([ctx[-1]] + drafts, pos)
                    j = 0
                    while j < len(drafts) and a[j] == drafts[j]:
                        j += 1
                    new = drafts[:j] + [a[j]]
                    ctx.extend(new)
                    backend.crop(len(ctx) - 1)      # last accepted token is not in KV yet
                    n_in, accepted = len(drafts) + 1, j + 1
                    if pacing is not None:
                        pacing.observe(accepted, len(drafts), tau_h + tau_t, lat)
        except StageLost as exc:
            if recovery is None:
                raise
            recovery(exc, ctx)                      # rebuilds to match ctx[:-1]
            continue

        if metrics:
            metrics.round(tau_h, tau_t, lat, n_in, accepted)

    # 投機一輪可以多吐幾個 token，所以會「超過」上限；對外要截掉，
    # 否則同一個 prompt 在 greedy 與 spec 回傳的長度不一樣，無法逐 token 對照。
    return ctx[len(prompt):len(prompt) + max_new_tokens]
