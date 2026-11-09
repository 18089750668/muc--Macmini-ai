"""一個最小的「多請求一起解碼」服務器：把 batch_decode 的槓桿接到真的請求上。

batch_decode 只證明「B 條一起跑，每步幾乎不變慢」。要把它變成使用者看得到的
東西，缺的是排程：一個人打字時 B 永遠是 1，只有**同時有多條請求**才湊得出 B。
所以這個檔案做的是「收請求 → 湊批 → 一起解碼 → 分別串流回去」。

三個設計決定：

1. **左補齊，不是右補齊。** 右補齊的話，最後一格是 pad，argmax 出來的是「pad 之後」
   的 token，直接錯；左補齊讓每條的真實 token 都結束在最後一格，argmax 的位置
   天然一致。代價是前面那些 pad 進了 KV，得用 per-row 遮罩擋掉（見 realmodel）。
2. **一個批次跑到底。** 某條先 EOS 了，它還留在批次裡繼續算（輸出丟掉），因為把它
   從批次中間抽掉就得重排 KV。這是刻意的簡化：浪費一點算力，換「絕不會錯位」。
3. **等一個窗口再開批**（`--window-ms`，預設 15ms）。等太久，單人延遲變差；
   等太短，湊不滿 B。這個值就是「用延遲換吞吐」的那個旋鈕。

    python -m cluster.serve --model-dir models/qwen2.5-0.5b --nodes 2 \
        --stage-port 7801 --head-port 7901 --port 8100

客戶端協議：一行一個 JSON。
    → {"prompt": "...", "max_tokens": 32}
    ← {"delta": " Paris"} ... {"done": true, "tokens": 32}
    ← {"error": "..."}（出錯時）
"""
import argparse
import json
import queue
import socket
import threading
import time

import mlx.core as mx
from tokenizers import Tokenizer

from . import protocol as P
from .coordinator import LocalBackend, RingHead, StageLost
from .draft import LookupDraft
from .main import Chain
from .pacing import Pacing
from .realmodel import RealConfig, RealModel


class Request:
    __slots__ = ("ids", "vf", "max_tokens", "out", "ids_out", "text", "done")

    def __init__(self, ids, max_tokens, out):
        self.ids = ids
        self.max_tokens = max_tokens
        self.out = out                 # queue.Queue，逐段文字丟給連線執行緒
        self.ids_out = []
        self.text = ""
        self.vf = 0


class Server:
    def __init__(self, backend, tok, eos, max_batch=8, window_ms=15.0,
                 kind="local", chain=None, head=None, mode="batch",
                 max_block=8, max_retries=3):
        self.backend = backend
        self.tok = tok
        self.eos = eos
        self.max_batch = max_batch
        self.window = window_ms / 1e3
        self.pending = queue.Queue()
        self.kind = kind
        self.chain, self.head = chain, head
        self.mode = mode                    # batch | spec | auto
        self.max_block = max_block
        self.max_retries = max_retries
        self.pacing = Pacing(max_block=max_block)
        self.drafter = LookupDraft(3)
        self.stats = {"requests": 0, "tokens": 0, "batches": 0, "losss": 0,
                      "replays": 0, "started": time.time()}

    # -- 收請求 ---------------------------------------------------------
    def handler(self, conn):
        f = conn.makefile("rwb")
        try:
            line = f.readline()
            if not line:
                return
            req_in = json.loads(line)
            prompt = req_in.get("prompt", "")
            ids = self.tok.encode(prompt).ids
            q = queue.Queue()
            req = Request(ids, int(req_in.get("max_tokens", 32)), q)
            self.pending.put(req)
            while True:
                piece = q.get()
                if piece is None:
                    break
                f.write(piece.encode() if isinstance(piece, bytes)
                        else json.dumps(piece).encode() + b"\n")
                f.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            # 客戶端自己走了（關視窗、斷網）。不要讓它影響這一輪的其他列。
            print(f"[serve] client went away: {type(exc).__name__}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:
                f.write(json.dumps({"error": str(exc)}).encode() + b"\n")
                f.flush()
            except Exception:
                pass
        finally:
            try:
                f.close()
                conn.close()
            except Exception:
                pass

    def accept_loop(self, port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(64)
        print(f"[serve] listening on :{port} "
              f"(max_batch={self.max_batch}, window={self.window*1e3:.0f}ms)", flush=True)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self.handler, args=(conn,), daemon=True).start()

    # -- 湊批 + 解碼 -----------------------------------------------------
    def _emit(self, req, toks_out):
        """把新 token 變成人看的文字；跨 token 邊界時只丟新增的那一段。"""
        req.ids_out.append(toks_out)
        new = self.tok.decode(req.ids_out)
        if new.startswith(req.text):
            delta = new[len(req.text):]
        else:                       # 極少數情況（merge 邊界被改寫）整段重送
            delta = new
        req.text = new
        if delta:
            req.out.put({"delta": delta})

    def _finish(self, req, reason):
        req.out.put({"done": True, "tokens": len(req.ids_out), "text": req.text,
                     "reason": reason})
        req.out.put(None)

    def _advance(self, req, tok):
        """推進一個 token 並回報是否結束（EOS 或到上限）。"""
        self._emit(req, tok)
        if tok == self.eos:
            self._finish(req, "eos")
            return True
        if len(req.ids_out) >= req.max_tokens:
            self._finish(req, "max_tokens")
            return True
        return False

    def _fail(self, req, exc):
        req.out.put({"error": f"cluster lost: {exc}"})
        req.out.put(None)

    # -- 批次：預填 → 逐輪解碼（可從「現在這一刻」重放續跑）-----------------
    def _batch_pass(self, active):
        """跑一輪批次。`active` 就地縮減（先 EOS 的從清單裡拿掉）。

        可重放的關鍵：每一列的完整上下文是 `ids + ids_out`，所以掉線之後不必
        從頭生成——重建鏈、把「現在的前綴」重跑一次 prefill，然後接著跑。
        付出的代價是重放前綴（O(context) 一次），不是丟掉已經算過的部分。
        """
        if not active:
            return
        self.stats["batches"] += 1
        ctxs = [r.ids + r.ids_out for r in active]
        L = max(len(c) for c in ctxs)
        rows = [[self.eos] * (L - len(c)) + c for c in ctxs]
        vf = [L - len(c) for c in ctxs]

        # 每個批次都是新的一組序列：KV 一定要清零，否則上一批的 batch 軸會被沿用
        self.backend.reset()
        toks, _, _, _ = self.backend.forward_rows(rows, 0, kind=P.PREFILL, valid_from=vf)
        self.stats["tokens"] += len(active)
        done = []
        for r, t, v in zip(active, toks, vf):
            r.vf = v
            if self._advance(r, t):
                done.append(r)
        for r in done:
            active.remove(r)
        if not active:
            return

        pos = L
        while active:
            cur = [[r.ids_out[-1]] for r in active]
            toks, _, _, _ = self.backend.forward_rows(
                cur, pos, kind=P.VERIFY, valid_from=[r.vf for r in active])
            self.stats["tokens"] += len(active)
            pos += 1
            done = []
            for r, t in zip(active, toks):
                if self._advance(r, t):
                    done.append(r)
            for r in done:
                active.remove(r)

    def _run_batch(self, batch):
        active = list(batch)
        tries = 0
        while True:
            try:
                self._batch_pass(active)
                return
            except StageLost as exc:
                tries += 1
                self.stats["losss"] += 1
                print(f"[serve] lost a stage mid-batch: {exc} "
                      f"(retry {tries}/{self.max_retries}, replay {len(active)} reqs)",
                      flush=True)
                ok = self._recover()
                if not ok or tries > self.max_retries:
                    for r in active:
                        self._fail(r, exc)
                    return
                self.stats["replays"] += 1

    # -- 單條投機：k 高就多送草稿，掉線一樣重放前綴 -------------------------
    def _spec_pass(self, req):
        be = self.backend
        ctx = list(req.ids) + list(req.ids_out)
        be.reset()
        a, th, tt, lat = be.step(ctx, 0, kind=P.PREFILL)
        self.stats["tokens"] += 1
        if self._advance(req, a[-1]):
            return

        while len(req.ids_out) < req.max_tokens:
            ctx = list(req.ids) + list(req.ids_out)
            pos = len(ctx) - 1                    # KV = len(ctx)-1 個位置
            blk = self.pacing.block()
            drafts = list(self.drafter.propose(ctx, blk))[:blk] if blk > 0 else []
            if not drafts:
                a, th, tt, lat = be.step([ctx[-1]], pos)
                self.pacing.observe(1, 0, th + tt, lat)
                self.stats["tokens"] += 1
                if self._advance(req, a[-1]):
                    return
                continue
            a, th, tt, lat = be.step([ctx[-1]] + drafts, pos)
            self.stats["tokens"] += 1
            j = 0
            while j < len(drafts) and a[j] == drafts[j]:
                j += 1
            self.pacing.observe(j + 1, len(drafts), th + tt, lat)
            new = drafts[:j] + [a[j]]
            be.crop(pos + 1 + j)      # 被拒的草稿佔過的 KV 要丟掉
            for t in new:
                if self._advance(req, t):
                    return

    def _run_one_spec(self, req):
        tries = 0
        while True:
            try:
                self._spec_pass(req)
                print(f"[serve] {self.pacing.line()}", flush=True)
                return
            except StageLost as exc:
                tries += 1
                self.stats["losss"] += 1
                print(f"[serve] lost a stage mid-request: {exc} "
                      f"(retry {tries}/{self.max_retries}, replaying prefix)", flush=True)
                ok = self._recover()
                if not ok or tries > self.max_retries:
                    self._fail(req, exc)
                    return
                self.stats["replays"] += 1

    def serve_forever(self, port):
        threading.Thread(target=self.accept_loop, args=(port,), daemon=True).start()
        while True:
            first = self.pending.get()
            batch, deadline = [first], time.perf_counter() + self.window
            while len(batch) < self.max_batch:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                try:
                    batch.append(self.pending.get(timeout=left))
                except queue.Empty:
                    break
            while True:              # 窗口裡可能還有人，一次撈乾淨
                try:
                    batch.append(self.pending.get_nowait())
                except queue.Empty:
                    break
                if len(batch) >= self.max_batch:
                    break
            self.stats["requests"] += len(batch)
            spec = self.mode == "spec" or (self.mode == "auto" and len(batch) == 1)
            try:
                if spec:
                    for r in batch:
                        self._run_one_spec(r)
                else:
                    self._run_batch(batch)
            except Exception as exc:      # 一個批次炸掉不能帶走整個服務
                import traceback
                traceback.print_exc()
                for r in batch:
                    if r.ids_out is not None:
                        try:
                            self._fail(r, f"{type(exc).__name__}: {exc}")
                        except Exception:
                            pass

    # -- 掉線：重建鏈。重建後由上層用「現在的前綴」重放，不從頭生成 -----------
    def _recover(self):
        if self.chain is None:
            print("[serve] backend is local; nothing to rebuild", flush=True)
            return False
        print("[serve] rebuilding the chain after a loss", flush=True)
        try:
            self.chain.rebuild()
            self.head.rebind(self.chain.out_sock, self.chain.in_sock)
            self.backend.reset()
            print("[serve] rebuilt", flush=True)
            return True
        except Exception as exc:
            print(f"[serve] rebuild failed: {exc}", flush=True)
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--stage-port", type=int, default=7801)
    ap.add_argument("--head-port", type=int, default=7901)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--window-ms", type=float, default=15.0)
    ap.add_argument("--mode", default="batch", choices=["batch", "spec", "auto"],
                    help="batch＝多條一起跑（吞吐）；spec＝單條投機（延遲）；auto＝有人一起就 batch")
    ap.add_argument("--max-block", type=int, default=8, help="pacing 的 block 上限")
    args = ap.parse_args()

    cfg = RealConfig(args.model_dir)
    tok = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")
    eos = tok.token_to_id("<|endoftext|>")

    chain = head = None
    if args.nodes >= 2:
        # 平均切層；最後一段回報 head（head 自己監聽 head 端口）
        n = cfg.n_layers
        step = n // args.nodes
        bounds = []
        lo = 0
        for i in range(args.nodes):
            hi = n if i == args.nodes - 1 else lo + step
            bounds.append((lo, hi))
            lo = hi
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", args.head_port))
        srv.listen(1)
        chain = Chain(cfg, bounds, [args.stage_port + i for i in range(args.nodes - 1)],
                      args.head_port, 0, srv, model_dir=args.model_dir)
        chain.start()
        head = RingHead(cfg, bounds[0][0], bounds[0][1], chain.out_sock,
                        chain.in_sock, model_dir=args.model_dir)
        backend = head
    else:
        backend = LocalBackend(cfg, model_dir=args.model_dir)

    srv = Server(backend, tok, eos, max_batch=args.max_batch,
                 window_ms=args.window_ms, chain=chain, head=head,
                 mode=args.mode, max_block=args.max_block)
    try:
        srv.serve_forever(args.port)
    except KeyboardInterrupt:
        pass
    finally:
        if chain is not None:
            chain.stop(abort=True)


if __name__ == "__main__":
    main()
