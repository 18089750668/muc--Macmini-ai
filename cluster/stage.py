"""Serve one contiguous block of layers. Receives a frame, runs the block,
either forwards it downstream or returns the argmaxes straight to the head."""
import os
import time

import mlx.core as mx
import numpy as np

from . import protocol as P
from .config import ToyConfig
from .fastmodel import FastModel
from .model import ToyModel, new_kv
from .realmodel import RealConfig, RealModel

DEBUG = os.environ.get("CLUSTER_DEBUG", "") not in ("", "0")


class StageServer:
    def __init__(self, cfg: ToyConfig, lo: int, hi: int, port: int,
                 next_addr=None, reply_to=None, fast=False, ctx_len=640,
                 persist=False, model_dir=None):
        # 真模型：這台只載入自己那幾層。第一段額外揹 embedding，最後一段額外
        # 揹 final norm + 輸出矩陣（tie 的時候那就是 embedding 本身，切不開）。
        self.real_dir = model_dir
        if model_dir:
            cfg = RealConfig(model_dir)
        self.cfg = cfg
        self.lo, self.hi = lo, hi
        self.port = port
        self.next_addr = next_addr
        self.reply_to = reply_to
        self.fast = FastModel(cfg, ctx_len) if fast else None
        if model_dir:
            self.model = RealModel(model_dir, lo=lo, hi=hi,
                                   need_embed=(lo == 0),
                                   need_head=(hi == cfg.n_layers))
        else:
            self.model = None if fast else ToyModel(cfg)
        self.kv = None if fast else new_kv(cfg.n_layers)
        # persist=True：這台是被家用戶手動拉起、要活好幾天的（會 remote publish
        # 出去），所以 ABORT 只結束這一輪，不結束進程。由 head 自己 spawn 的
        # stage 保持 persist=False，head 一收工就退，避免殘留佔埠。
        self.persist = persist

    def _forward(self, arr, start_pos, meta=None):
        """[B,L,H] 或舊呼叫的 [L,H]（視為 B=1）→ [B,L,H]。

        batching 之後 hidden 一定是 3 維；舊的 2 維呼叫保留，是為了不打破既有
        的單序列路徑與回歸測試。
        """
        if self.fast is None:
            h = mx.array(arr)
            if h.ndim == 2:
                h = h[None]
            meta = meta or {}
            if DEBUG and meta.get("valid_from"):
                print(f"[stage {self.lo}:{self.hi}] valid_from={meta['valid_from']}",
                      flush=True)
            vf = meta.get("valid_from")
            if vf is not None:
                vfa = mx.array(vf, mx.int32)
                # 內容位置可以由 start_pos / L / valid_from 推出來，不必上線傳
                rp = (mx.arange(start_pos, start_pos + h.shape[1])[None, :]
                      - vfa[:, None])
            else:
                vfa, rp = None, None
            return self.model.forward_range(h, start_pos, self.lo, self.hi, self.kv,
                                            rope_pos=rp, valid_from=vfa)
        if getattr(arr, "ndim", 2) == 3:
            raise SystemExit("[stage] --fast 路徑還沒有 batch 版本；batch 走 realmodel")
        x = mx.array(arr)[None]
        if arr.shape[0] == 1:
            return self.fast.step(x[0], mx.array(start_pos, mx.int32),
                                  self.lo, self.hi)[None]
        return self.fast.forward_batch(x, start_pos, self.lo, self.hi)

    def _logits(self, out_h):
        if self.fast is None:
            return self.model.logits_all(out_h)
        lg, hn = self.fast.logits(out_h[0])      # [L, V] -> [1, L, V] to match
        return lg[None], hn[None]

    def serve(self):
        import socket
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(1)
        print(f"[stage] layers[{self.lo}:{self.hi}] listening :{self.port}", flush=True)

        # 一個 stage 可能活很久（家用機器開機就常駐），上游卻會斷：筆記本闔蓋、
        # 換 Wi-Fi、head 重啟。所以這裡不是「上游斷了就自殺」，而是回到 accept
        # 等下一次連線 -- 只有明確的 ABORT 才結束。
        while True:
            conn, _ = srv.accept()
            P.tune(conn)
            # 新上游 = 新回合。head 重連後會重放整個前綴，所以這裡把 KV 清掉，
            # 不讓上一條連線留下的殘留位置把 token 悄悄算錯。
            if self.fast is not None:
                self.fast = FastModel(self.cfg, self.fast.ctx)
            else:
                self.kv = new_kv(self.cfg.n_layers)
            out = P.connect(*self.next_addr, retries=200) if self.next_addr else None
            reply = P.connect(*self.reply_to, retries=200) if self.reply_to else None
            aborted = False
            try:
                aborted = self._loop(conn, out, reply)
            except (ConnectionError, BrokenPipeError, OSError):
                # upstream died (killed or rebuilt): the host notices on its own
                # side and replays the prefix, so there is nothing to repair here.
                nxt = "waiting for a new upstream" if self.persist else "exiting"
                print(f"[stage {self.lo}:{self.hi}] upstream gone, {nxt}",
                      flush=True)
                if not self.persist:
                    # head spawn 出來的 stage：上游沒了就退，否則它會一直佔著
                    # 埠等一個不會再來的人，把重建時的新 stage 擋在門外。
                    return
            finally:
                for sock in (conn, out, reply):
                    if sock is not None:
                        try:
                            sock.close()
                        except OSError:
                            pass
            if aborted and not self.persist:
                return

    def _loop(self, conn, out, reply):
        while True:
            P.arm(conn)
            ftype, meta, arr = P.recv_frame(conn)
            if ftype == P.ABORT:
                print("[stage] abort", flush=True)
                if out is not None:
                    try:
                        P.send_frame(out, P.ABORT, {})
                    except OSError:
                        pass
                return True
            if ftype == P.RESET:
                if self.fast is not None:
                    self.fast = FastModel(self.cfg, self.fast.ctx)
                else:
                    self.kv = new_kv(self.cfg.n_layers)
                if out is not None:
                    P.send_frame(out, P.RESET, {})
                continue
            if ftype == P.HEARTBEAT:
                # 只是「你還在嗎」。不計任何東西，像 PREFILL 一樣往下傳：
                # 頭在 in_sock 上等，所以心跳必須整條鏈走完一圈才算數 --
                # 順便量到的就是整條鏈的往返，不是某一跳。
                down = out if out is not None else reply
                P.send_frame(down, P.HEARTBEAT, dict(meta or {}), None)
                continue
            if ftype not in (P.PREFILL, P.VERIFY):
                print(f"[stage] ignoring {ftype}", flush=True)
                continue

            t0 = time.perf_counter()
            out_h = self._forward(arr, meta["start_pos"], meta)
            mx.eval(out_h)
            tau = (time.perf_counter() - t0) * 1e3

            if reply is not None:
                lg, _ = self._logits(out_h)
                mx.eval(lg)
                # B=1 時維持原本行為：每個位置一個 argmax（投機解碼的 chunk 驗證
                # 要靠這個）。B>1 是批次解碼，每列只取最後一個位置。
                if lg.shape[0] == 1:
                    toks = mx.argmax(lg[0], axis=-1).tolist()
                else:
                    toks = mx.argmax(lg, axis=-1)[:, -1].tolist()
                if DEBUG:
                    print(f"[stage {self.lo}:{self.hi}] in{arr.shape} out{out_h.shape} "
                          f"logits{lg.shape} toks{len(toks)}", flush=True)
                P.send_frame(reply, P.RESULT, {
                    "req": meta.get("req", 0),
                    "chunk_id": meta.get("chunk_id", 0),
                    "start_pos": meta["start_pos"],
                    "batch": int(lg.shape[0]),
                    "n_in": int(out_h.shape[1]),
                    "tau_ms": tau,
                }, np.asarray(toks, dtype=np.int32))
            elif out is not None:
                if DEBUG:
                    print(f"[stage {self.lo}:{self.hi}] in{arr.shape} fwd{out_h.shape}",
                          flush=True)
                P.send_frame(out, ftype, dict(meta), np.array(out_h))
            else:
                raise RuntimeError("stage has neither next nor reply")
