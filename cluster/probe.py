"""Measure this machine, then feed the numbers to the planner.

A stage is not `n * tau`, it is `fixed + n * tau`. Both matter and both are
machine-specific, so neither may be hand-filled. This probes them under the
REALISTIC condition -- decode with a growing KV, not a fixed shape loop -- because
fixed-shape benchmarking understates the live cost by ~4x (see RESULTS.md).

Writes profile.json:  {"tau_ms_per_layer":..,"stage_fixed_ms":..,"n_layers":..}
"""
import json
import statistics
import sys
import time

import mlx.core as mx

from .config import ToyConfig
from .fastmodel import FastModel
from .model import ToyModel, new_kv


def seg_time(model, cfg, L, reps=15, base_pos=48, fast=False):
    """One stage of L layers, decoding token by token with KV growing each step."""
    if fast:
        fm = FastModel(cfg, max(64, base_pos + reps + 16))
        fm.forward_batch(fm.embed([1] * base_pos), 0, 0, L)
        mx.eval(mx.zeros((1,)))
        ts = []
        for i in range(reps):
            t0 = time.perf_counter()
            out = fm.step(fm.embed_one(1), mx.array(base_pos + i, mx.int32), 0, L)
            mx.eval(out)
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    kv = new_kv(cfg.n_layers)
    h = model.embed([1] * base_pos)
    model.forward_range(h, 0, 0, L, kv)
    mx.eval(mx.zeros((1,)))
    ts = []
    for i in range(reps):
        h = model.embed([1])
        t0 = time.perf_counter()
        out = model.forward_range(h, base_pos + i, 0, L, kv)
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def run(quiet=False, fast=False):
    cfg = ToyConfig()
    model = ToyModel(cfg)
    shapes = {L: seg_time(model, cfg, L, fast=fast) for L in (1, 2, 4, 8)}
    if not quiet:
        print("每段層數 -> 該段耗時（活迴圈、KV 逐 token 增長，"
              + ("FastModel + compile）" if fast else "ToyModel 串接）"))
        for L, t in shapes.items():
            print(f"  {L} 層: {t:.3f} ms")
    t1, t8 = shapes[1], shapes[8]
    tau = (t8 - t1) / 7.0
    fixed = t1 - tau
    prof = dict(tau_ms_per_layer=round(tau, 4),
                stage_fixed_ms=round(fixed, 4),
                n_layers=cfg.n_layers,
                decode="fastmodel" if fast else "toymodel",
                samples={str(k): round(v, 4) for k, v in shapes.items()})
    if not quiet:
        print(f"\n  tau   = {tau:.3f} ms/層")
        print(f"  fixed = {fixed:.3f} ms/段（每多一段就多付一次）")
    return prof


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    prof = run(fast=("--fast" in argv))
    out = "profile.json"
    if "--write" in argv:
        with open(out, "w") as f:
            json.dump(prof, f, indent=2)
        print(f"\n已寫入 {out}；`main plan --profile {out}` 可直接吃這個檔")


if __name__ == "__main__":
    main()
