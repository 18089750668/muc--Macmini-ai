"""M0: is this workload launch-bound on an M1?

Measures, in order:
  1. single-op launch overhead (sync every op) -> the floor per kernel
  2. per-layer time for block sizes 1..8 at L=1 (decode shape)
  3. single-machine decode tok/s for the toy model

If launch overhead dominates the per-layer time, no distributed scheme and no
faster machine will help -- fuse/compile first.
"""
import statistics
import time

import mlx.core as mx

from cluster.config import ToyConfig
from cluster.model import ToyModel, new_kv


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
        mx.eval(mx.zeros((1,)))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.eval(mx.zeros((1,)))
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), min(times)


def main():
    cfg = ToyConfig()
    model = ToyModel(cfg)
    H = cfg.hidden

    print(f"device={mx.default_device()}  layers={cfg.n_layers} hidden={H}")
    print(f"mem={mx.get_active_memory() / 1e6:.1f}MB active\n")

    x = mx.zeros((1, H))
    launch_ms, launch_min = bench(lambda: mx.add(x, x))
    print(f"1. single-op launch : median {launch_ms * 1e3:.1f}us  min {launch_min * 1e3:.1f}us")

    print("\n2. per-layer (L=1 decode shape)")
    print(f"{'layers':>7} {'tau_ms':>9} {'ms/layer':>10} {'us/layer':>10}")
    for n in (1, 2, 4, 8):
        kv = new_kv(cfg.n_layers)
        h = mx.zeros((1, 1, H))
        tau, _ = bench(lambda: model.forward_range(h, 0, 0, n, kv), warmup=5, iters=30)
        print(f"{n:>7} {tau:>9.3f} {tau / n:>10.3f} {tau / n * 1e3:>10.1f}")

    print("\n3. single-machine decode")
    kv = new_kv(cfg.n_layers)
    tok = mx.array([[3]], mx.int32)
    t0 = time.perf_counter()
    for i in range(64):
        h = model.embed([3])
        h = model.forward_range(h, i, 0, cfg.n_layers, kv)
        lg, _ = model.logits_all(h)
        nxt = mx.argmax(lg[0, -1])
        mx.eval(nxt)
        tok = nxt.reshape(1, 1)
    wall = time.perf_counter() - t0
    print(f"  64 tokens: {wall * 1e3:.1f}ms  -> {64 / wall:.2f} tok/s")

    print("\n審判：把 ms/layer 和 launch 對照。若每層的 op 數 × launch > tau 的一半，")
    print("就是 launch-bound；先做 mx.compile / 算子融合，再談分機。")


if __name__ == "__main__":
    main()
