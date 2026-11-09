"""Is `mx.compile` + fixed KV slab the fix for 0.38ms/层?

Two things must be true at once:
  1. it must be FASTER (otherwise the complexity is not paid for)
  2. it must produce the SAME tokens (otherwise "faster" is meaningless)

So this checks correctness first, then measures. Runs against the existing
ToyModel in the same process, same weights, same prompt.
"""
import sys
import time

import mlx.core as mx
import numpy as np

from cluster.config import ToyConfig
from cluster.fastmodel import FastModel
from cluster.model import ToyModel, new_kv
from cluster.main import make_prompt, token_hash

PROMPT, GEN = 512, 32
SLAB = PROMPT + GEN + 64


def baseline_greedy(cfg):
    m = ToyModel(cfg)
    kv = new_kv(cfg.n_layers)
    prompt = make_prompt(cfg, PROMPT)
    ctx = list(prompt)
    h = m.embed(ctx)
    h = m.forward_range(h, 0, 0, cfg.n_layers, kv)
    lg, _ = m.logits_all(h)
    mx.eval(lg)
    ctx.append(int(mx.argmax(lg[0, -1])))
    while len(ctx) - len(prompt) < GEN:
        h = m.embed([ctx[-1]])
        h = m.forward_range(h, len(ctx) - 1, 0, cfg.n_layers, kv)
        lg, _ = m.logits_all(h)
        mx.eval(lg)
        ctx.append(int(mx.argmax(lg[0, -1])))
    out = ctx[len(prompt):]
    return out, token_hash(out)


def fast_greedy(cfg, fm=None):
    prompt = make_prompt(cfg, PROMPT)
    ctx = list(prompt)
    fm = FastModel(cfg, SLAB)
    fm.prefill(ctx[:-1], ctx_cap=SLAB)
    x = fm.embed_one(ctx[-1])
    h = fm.step(x, mx.array(len(ctx) - 1, mx.int32))
    lg, _ = fm.logits(h)
    mx.eval(lg)
    ctx.append(int(mx.argmax(lg[0])))
    while len(ctx) - len(prompt) < GEN:
        x = fm.embed_one(ctx[-1])
        h = fm.step(x, mx.array(len(ctx) - 1, mx.int32))
        lg, _ = fm.logits(h)
        mx.eval(lg)
        ctx.append(int(mx.argmax(lg[0])))
    out = ctx[len(prompt):]
    return out, token_hash(out), fm


def segment_profile(cfg, fm, shapes=(1, 2, 4, 8), reps=15, base=64, compile=True):
    """Stage cost per layer count, measured in the live (growing) loop."""
    res = {}
    for L in shapes:
        fm = FastModel(cfg, SLAB, compile=compile)
        for p in range(base):                        # prime the slab
            fm.step(fm.embed_one(1), mx.array(p, mx.int32), 0, L)
        mx.eval(fm.ks, fm.vs)
        ts = []
        for i in range(reps):
            x = fm.embed_one(1)
            t0 = time.perf_counter()
            h = fm.step(x, mx.array(base + i, mx.int32), 0, L)
            mx.eval(h)
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        res[L] = ts[len(ts) // 2]
    return res


def main():
    cfg = ToyConfig()
    prompt = make_prompt(cfg, PROMPT)

    t0 = time.perf_counter()
    tok_b, h_b = baseline_greedy(cfg)
    t_base = time.perf_counter() - t0

    fm = FastModel(cfg, SLAB)
    tok_f, h_f, _ = fast_greedy(cfg, fm)

    print(f"baseline  hash={h_b}  ({t_base:.2f}s, 含 prefill)")
    print(f"compiled  hash={h_f}")
    ok = h_b == h_f
    print(f"[正確性] {'一致' if ok else '不一致'}")
    if not ok:
        diff = [i for i, (a, b) in enumerate(zip(tok_b, tok_f)) if a != b]
        print(f"  首個分歧位置: {diff[:3]}  base={[tok_b[i] for i in diff[:3]]} "
              f"fast={[tok_f[i] for i in diff[:3]]}")
        return 1

    print("\n每段層數 -> 段耗時（活迴圈，KV slab 固定 + compile）")
    res = segment_profile(cfg, fm, base=64)
    old = {1: 0.657, 2: 1.064, 4: 1.896, 8: 3.334}
    for L, t in res.items():
        o = old.get(L)
        print(f"  {L} 層: {t:.3f} ms" +
              (f"   (之前 {o:.3f} ms, {o / t:.2f}×)" if o else ""))

    print("\n不 compile、只用固定 slab（隔離 compile 的作用）")
    res2 = segment_profile(cfg, fm, base=64, compile=False)
    for L, t in res2.items():
        print(f"  {L} 層: {t:.3f} ms   (之前 0.657/1.064/1.896/3.334)")
    t1, t8 = res[1], res[8]
    tau, fixed = (t8 - t1) / 7.0, t1 - (t8 - t1) / 7.0
    print(f"\n  tau   = {tau:.3f} ms/層  (之前 0.383)")
    print(f"  fixed = {fixed:.3f} ms/段  (之前 0.274)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
