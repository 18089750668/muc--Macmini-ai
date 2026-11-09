"""CLI: single | run | stage | plan"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

from . import protocol as P
from .config import ToyConfig
from .coordinator import LocalBackend, RingHead, generate
from .metrics import Metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_range(s):
    lo, hi = s.split(":")
    return int(lo), int(hi)


def make_prompt(cfg, length):
    rng = np.random.default_rng(7)
    return rng.integers(0, cfg.vocab, size=length).tolist()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]


def report(name, tokens, metrics, wall):
    s = metrics.summary(wall, len(tokens))
    print(f"\n[{name}]")
    print(f"  tokens={len(tokens)} hash={token_hash(tokens)}")
    print(f"  tok/s={s['tok_s']:.2f}  rounds={s['rounds']}")
    print(f"  tavg={s['tavg_ms']:.2f}ms  round_latency={s['latency_ms']:.2f}ms")
    if s.get("prefill_ms"):
        print(f"  prefill={s['prefill_ms']:.0f}ms")
    print(f"  tau_head={s['tau_h_ms']:.2f}ms  tau_tail={s['tau_t_ms']:.2f}ms"
          f"  hop={s['hop_ms']:.2f}ms")
    print(f"  g(accepted/round)={s['g']:.2f}  frames/token={s['frames_per_token']:.2f}"
          f"  useful_inflight={s['useful_inflight']:.2f}  max_inflight={s['max_inflight']}")
    return s


def cmd_single(args):
    cfg = ToyConfig()
    prompt = make_prompt(cfg, args.prompt_len)
    best, best_tokens = None, None
    for _ in range(args.repeat):
        backend = LocalBackend(cfg)
        m = Metrics()
        t0 = time.perf_counter()
        tokens = generate(backend, prompt, args.max_tokens, block=args.block,
                          mode=args.mode, metrics=m)
        wall = time.perf_counter() - t0
        s = report("single", tokens, m, wall)
        if best is None or s["tok_s"] > best["tok_s"]:
            best, best_tokens = s, tokens
        if _ < args.repeat - 1:
            continue
    print(f"\n[best-of-{args.repeat}] tok/s={best['tok_s']:.2f}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"tokens": best_tokens, **best}, f, indent=2)


class Chain:
    """Owns the stage processes and the two sockets the head talks through.

    Rebuilding everything is deliberate: the head is the only node holding the
    authoritative token prefix, so the cheapest CORRECT recovery is to respawn
    the whole chain and replay that prefix. Keeping survivors alive buys a few
    hundred milliseconds and risks silent KV skew -- the failure mode that
    produces wrong tokens with no error at all.
    """

    def __init__(self, cfg, bounds, ports, head_port, spin_us, srv, fast=False,
                 peers=None, advertise=None, model_dir=None):
        self.cfg, self.bounds, self.ports = cfg, list(bounds), list(ports)
        self.head_port, self.spin_us, self.srv = head_port, spin_us, srv
        self.fast = fast
        if isinstance(peers, str):
            peers = [x.strip() for x in peers.split(",") if x.strip()]
        self.peers = peers or None   # ["host:port", ...] stages already running elsewhere
        self.advertise = advertise  # "host:port" the stages use to reach this head
        self.out_sock = self.in_sock = None
        self.procs = []
        self.builds = 0
        self.model_dir = model_dir

    def _check_ports(self):
        """Fail loudly if a port is already taken by a leftover process.

        A stale stage still listening means the head connects to a machine holding
        DIFFERENT layers: every shape stays legal, nothing crashes, and the output
        is silently wrong. Caught this live once (hash fbb3884b4293cdd0 instead of
        126227e02636ad3a, and faster than possible because one stage did no work).
        """
        import socket as _s
        taken = []
        for pt in self.ports[:len(self.bounds) - 1]:
            probe = _s.socket()
            # 這裡必須帶 SO_REUSEADDR：不帶的話，**只剩 TIME_WAIT 的連線**也會
            # 讓 bind 失敗，於是「剛關掉一條鏈、馬上重建」會被自己的預檢擋下來
            # （實測：bench 反覆重建鏈時 Errno 48，但 lsof 上根本沒有 LISTEN）。
            # 帶上之後語意才是對的：TIME_WAIT 可 reuse，真的有人 LISTEN 仍會擋。
            probe.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", pt))
            except OSError:
                taken.append(pt)
            finally:
                probe.close()
        if taken:
            raise SystemExit(
                f"[host] 端口被佔用：{taken}（多半是上一次沒退乾淨的 stage）。\n"
                f"        先 lsof -nP -iTCP:<port> -sTCP:LISTEN -t | xargs kill 換掉，"
                f"或改 --stage-port 到另一個區段。\n"
                f"        不修就硬跑，結果會是『合法形狀 + 錯誤 token』，不會 crash。")

    def spawn(self):
        self.procs = []
        if self.peers:                 # remote stages were started by the operator
            return
        self._check_ports()
        reply = self.advertise or f"127.0.0.1:{self.head_port}"
        n_stage = len(self.bounds) - 1
        for i in range(n_stage):
            lo, hi = self.bounds[i + 1]
            cmd = [sys.executable, "-m", "cluster.main", "stage",
                   "--layers", f"{lo}:{hi}", "--port", str(self.ports[i]),
                   "--spin-us", str(self.spin_us)] + (["--fast"] if self.fast else [])
            if self.model_dir:
                cmd += ["--model-dir", self.model_dir]
            if i + 1 < n_stage:
                cmd += ["--next", f"127.0.0.1:{self.ports[i + 1]}"]
            else:
                cmd += ["--reply", reply]
            self.procs.append(subprocess.Popen(cmd, cwd=ROOT))

    def connect(self):
        if self.peers:
            host, port = self.peers[0].rsplit(":", 1)
            self.out_sock = P.connect(host, int(port), retries=200)
        else:
            self.out_sock = P.connect("127.0.0.1", self.ports[0], retries=200)
        self.srv.settimeout(90)
        self.in_sock, _ = self.srv.accept()
        self.in_sock.settimeout(90)

    def start(self):
        self.builds += 1
        self.spawn()
        self.connect()

    def stop(self, abort=False):
        if abort and self.out_sock is not None:
            try:
                P.send_frame(self.out_sock, P.ABORT, {})
            except Exception:
                pass
        for name in ("out_sock", "in_sock"):
            s = getattr(self, name)
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
                setattr(self, name, None)
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []
        # 監聽 socket 也要關。只關上下游會讓 head_port 留在 LISTEN，之後在
        # 同一個端口重建鏈就會 Errno 48：那不是 TIME_WAIT，是真的還有人佔著。
        if getattr(self, "srv", None) is not None:
            try:
                self.srv.close()
            except Exception:
                pass
            self.srv = None

    def _open_srv(self):
        """（重建用）重新開一個監聽 socket。

        `stop()` 會把 `srv` 關掉並設成 None——這是對的（不關就 Errno 48）。
        但這樣一來 `rebuild()` 直接呼叫 `start()` 就會在 `connect()` 裡撞到
        `self.srv.settimeout()` → NoneType。所以重建前必須自己再開一個。
        """
        import socket as _s
        srv = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        srv.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.head_port))
        srv.listen(1)
        self.srv = srv

    def rebuild(self, bounds=None, ports=None):
        self.stop(abort=False)
        if getattr(self, "srv", None) is None:
            self._open_srv()
        if bounds is not None:
            self.bounds = list(bounds)
        if ports is not None:
            self.ports = list(ports)
        self.start()


def cmd_run(args):
    md = getattr(args, "model_dir", None)
    if md:
        from .realmodel import RealConfig
        cfg = RealConfig(md)
    else:
        cfg = ToyConfig()
    P.set_spin(args.spin_us)
    from .planner import plan, validate, describe

    spare = getattr(args, "spare", 0)
    if args.nodes:
        workers = max(1, args.nodes - spare)
        if spare:
            print(f"[host] {args.nodes} 台 = {workers} 台工作 + {spare} 台热备", flush=True)
        bounds, _cost, _info = plan([1.0] * cfg.n_layers, workers,
                                    hop_ms=args.hop_ms, objective=args.objective)
        if bounds is None:
            raise SystemExit("planner found no feasible tiling; run `plan` for details")
        issues = validate([1.0] * cfg.n_layers, bounds)
        print(f"[host] 自我检测: {'通过' if not issues else issues}", flush=True)
        ports = [args.stage_port + i for i in range(len(bounds) - 1)]
    else:
        head_lo, head_hi = parse_range(args.layers)
        bounds = [(head_lo, head_hi), (head_hi, cfg.n_layers)]
        ports = [int(args.tail.split(":")[1])]

    head_lo, head_hi = bounds[0]
    head_port = args.head_port
    print(f"[plan] {len(bounds)} segments: {describe(None, bounds)}", flush=True)

    prompt = make_prompt(cfg, args.prompt_len)
    chain = None
    if len(bounds) == 1:
        # The planner's own answer can be "do not distribute": under objective=sum
        # every extra hop costs more than the layers it removes. Honour it.
        print("[host] 單節點：planner 判定加機器純虧，全層本地跑", flush=True)
        backend = LocalBackend(cfg, model_dir=md)
    else:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", head_port))
        srv.listen(1)
        chain = Chain(cfg, bounds, ports, head_port, args.spin_us, srv,
                  fast=args.fast,
                  peers=getattr(args, "peers", None),
                  advertise=getattr(args, "advertise", None),
                  model_dir=md)
        chain.start()
        backend = RingHead(cfg, head_lo, head_hi, chain.out_sock, chain.in_sock,
                           fast=args.fast, ctx_len=args.ctx_len, model_dir=md)
    losses = {"n": 0}

    def on_loss(exc, ctx):
        """Host judgement: is this recoverable, and does the plan change?"""
        losses["n"] += 1
        print(f"\n[host] 掉线: {exc}", flush=True)
        if chain is None:
            raise exc
        if args.on_loss == "abort":
            print("[host] 判定: policy=abort，整体失败", flush=True)
            raise exc
        if losses["n"] > args.max_recover:
            print(f"[host] 判定: 已恢复 {args.max_recover} 次，放弃（可能是系统性问题）",
                  flush=True)
            raise exc
        print(f"[host] 判定: 可恢复 {losses['n']}/{args.max_recover}  policy={args.on_loss}"
              f"  已知前缀={len(ctx)} tokens  重建次数={chain.builds + 1}", flush=True)
        if args.on_loss == "shrink":
            w = len(chain.bounds) - 1
            if w <= 1:
                print("[host] 只剩一个工作节点，无法再缩，改为原样重建", flush=True)
                chain.rebuild()
            else:
                nb, _c, _i = plan([1.0] * cfg.n_layers, w, hop_ms=args.hop_ms,
                                  objective=args.objective)
                if nb is None:
                    chain.rebuild()
                else:
                    print(f"[host] 重新分配 {w} 台: {describe(None, nb)}", flush=True)
                    chain.rebuild(bounds=nb, ports=[args.stage_port + i
                                                    for i in range(len(nb) - 1)])
                    backend.lo, backend.hi = nb[0]
        else:
            chain.rebuild()
        backend.rebind(chain.out_sock, chain.in_sock)
        t0 = time.perf_counter()
        backend.replay(ctx[:-1])
        print(f"[host] 重放 {len(ctx) - 1} tokens 重建 KV 完成: "
              f"{time.perf_counter() - t0:.2f}s，继续生成", flush=True)

    best, best_tokens = None, None
    n_stage = len(bounds) - 1
    try:
        for rep in range(args.repeat):
            m = Metrics()
            if rep > 0:
                backend.reset()
                backend.crop(0)
            t0 = time.perf_counter()
            tokens = generate(backend, prompt, args.max_tokens, block=args.block,
                              mode=args.mode, metrics=m, recovery=on_loss)
            wall = time.perf_counter() - t0
            s = report(f"ring[{bounds[0][0]}:{bounds[0][1]}] +{n_stage} stages",
                       tokens, m, wall)
            s["recoveries"] = losses["n"]
            s["builds"] = chain.builds if chain else 0
            if best is None or s["tok_s"] > best["tok_s"]:
                best, best_tokens = s, tokens
            if losses["n"]:
                print(f"  recoveries={losses['n']} builds={s['builds']}", flush=True)
    finally:
        if chain is not None:
            chain.stop(abort=True)

    print(f"\n[best-of-{args.repeat}] tok/s={best['tok_s']:.2f}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"tokens": best_tokens, **best}, f, indent=2)


def cmd_stage(args):
    lo, hi = parse_range(args.layers)
    md = getattr(args, "model_dir", None)
    cfg = ToyConfig()
    if md:
        from .realmodel import RealConfig
        cfg = RealConfig(md)
    P.set_spin(args.spin_us)
    next_addr = None
    reply_to = None
    if args.reply:
        host, port = args.reply.split(":")
        reply_to = (host, int(port))
    if args.next:
        host, port = args.next.split(":")
        next_addr = (host, int(port))
    from .stage import StageServer
    StageServer(cfg, lo, hi, args.port, next_addr=next_addr, reply_to=reply_to,
                fast=args.fast, ctx_len=args.ctx_len,
                persist=args.persist, model_dir=md).serve()


def cmd_plan(args):
    from .planner import plan, min_nodes, max_ctx, validate, describe
    cfg = ToyConfig()
    n_layers = cfg.n_layers
    mem = None
    if args.model:
        from .footprint import MODELS
        m = next((x for x in MODELS if x.name.startswith(args.model)), None)
        if m is None:
            raise SystemExit(f"unknown model {args.model}")
        n_layers = m.n_layers
    workers = max(1, args.nodes - getattr(args, "spare", 0))
    if getattr(args, "spare", 0):
        print(f"[4台/自檢] {args.nodes} 台 = {workers} 台工作 + {args.spare} 台熱備")
    if args.model:
        mem = dict(layer_w=m.weight_bytes(1),
                   layer_kv_token=m.kv_bytes_per_token(1),
                   ctx=args.ctx, head=m.head_bytes(),
                   capacity=[args.capacity * 1e9] * workers)

    costs = [args.layer_ms] * n_layers
    fixed = args.stage_fixed_ms
    if args.profile:
        with open(args.profile) as f:
            prof = json.load(f)
        tau = prof["tau_ms_per_layer"]
        fixed = prof["stage_fixed_ms"]
        costs = [tau] * n_layers
        print(f"[profile] tau={tau}ms/層  fixed={fixed}ms/段  (來自 {args.profile})")

    b, cost, info = plan(costs, workers, args.hop_ms, None, args.objective,
                         fixed, mem)
    if b is None:
        print(f"不可行：{args.nodes} 台（{workers} 台工作）× {args.capacity:.0f}GB 放不下")
        k, b2, c2 = min_nodes(costs, n_layers, args.hop_ms, None, fixed,
                              dict(mem, capacity=[args.capacity * 1e9] * n_layers)
                              if mem else None, args.objective)
        if b2:
            print(f"[自檢] 最少需要 {k} 台: {describe(None, b2)}  cost={c2:.2f}ms")
        if mem:
            mx = max_ctx(costs, args.nodes, mem, args.objective, args.hop_ms, fixed)
            if mx:
                print(f"[自檢] {args.nodes} 台能跑的最大上下文: {mx // 1024}K"
                      f"（每台 {args.capacity:.0f}GB 可用）")
            else:
                print(f"[自檢] {args.nodes} 台連 1K 上下文都放不下")
        return

    issues = validate(costs, b, mem)
    print(describe(None, b))
    print(f"used {len(b)}/{workers} 工作節點  cost={cost:.2f}ms  "
          f"objective={args.objective}  hop={args.hop_ms}ms  fixed={fixed}ms"
          + (f"  peak={info['peak_bytes'] / 1e9:.1f}G" if mem else ""))
    print(f"[自檢] {'通過：層區間連續、無空洞、每台放得下' if not issues else issues}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cluster")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("single", cmd_single), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--max-tokens", type=int, default=32)
        p.add_argument("--prompt-len", type=int, default=16)
        p.add_argument("--block", type=int, default=4)
        p.add_argument("--mode", choices=["greedy", "spec"], default="greedy")
        p.add_argument("--repeat", type=int, default=1)
        p.add_argument("--json-out", default=None)
        if name == "run":
            p.add_argument("--layers", default="0:4")
            p.add_argument("--tail", default="127.0.0.1:5201")
            p.add_argument("--nodes", type=int, default=0,
                           help="auto-plan N nodes with the DP planner (ignores --layers/--tail)")
            p.add_argument("--stage-port", type=int, default=5201)
            p.add_argument("--hop-ms", type=float, default=0.05)
            p.add_argument("--objective", choices=["max", "sum"], default="max",
                           help="max = stages run concurrently; sum = plain relay chain")
            p.add_argument("--head-port", type=int, default=5200)
            p.add_argument("--spin-us", type=int, default=0,
                           help="poll window in us before sleeping (0 = always sleep)")
            p.add_argument("--spare", type=int, default=0,
                           help="nodes kept as hot stand-by (4 = 3 workers + 1 spare)")
            p.add_argument("--on-loss", choices=["rebuild", "shrink", "abort"],
                           default="rebuild",
                           help="host policy when a node dies: rebuild the chain, "
                                "shrink onto fewer nodes, or fail")
            p.add_argument("--peers", default="",
                           help="comma list host:port of already-running stages "
                                "(first is the one the head talks to); skips spawning")
            p.add_argument("--advertise", default="",
                           help="host:port stages use to reach this head "
                                "(needed when the head is on another machine)")
            p.add_argument("--model-dir", default=None,
                           help="真模型目錄（config.json + model.safetensors）")
            p.add_argument("--fast", action="store_true",
                           help="compiled fixed-shape decode (FastModel)")
            p.add_argument("--ctx-len", type=int, default=640,
                           help="KV slab length when --fast")
            p.add_argument("--max-recover", type=int, default=3,
                           help="give up after this many recoveries in one run")
        p.set_defaults(func=fn)

    p = sub.add_parser("stage")
    p.add_argument("--layers", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--reply", default=None)
    p.add_argument("--next", default=None)
    p.add_argument("--spin-us", type=int, default=0)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--ctx-len", type=int, default=640)
    p.add_argument("--model-dir", default=None, help="真模型目錄")
    p.add_argument("--persist", action="store_true",
                   help="ABORT 之後不退出，等下一次上游連線（家用常駐 stage 用這個）")
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser("plan")
    p.add_argument("--nodes", type=int, default=2)
    p.add_argument("--layer-ms", type=float, default=1.0)
    p.add_argument("--hop-ms", type=float, default=0.2)
    p.add_argument("--objective", choices=["max", "sum"], default="max")
    p.add_argument("--stage-fixed-ms", type=float, default=0.0,
                   help="per-stage constant (measured ~0.42ms on M1 toy model)")
    p.add_argument("--model", default=None, help="8B | 70B | 200B (adds memory constraint)")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--capacity", type=float, default=24.0, help="usable GB per node")
    p.add_argument("--profile", default=None, help="profile.json from `cluster.probe`")
    p.add_argument("--spare", type=int, default=0,
                   help="of --nodes, how many stay as hot stand-by")
    p.set_defaults(func=cmd_plan)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
