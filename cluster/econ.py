"""The money question: given a model that does NOT fit on one machine, what does
each extra machine buy you -- capacity, or speed, or neither?

This exists because "more machines is cheaper" and "more machines is slower" are
both true, about different things:

  * CAPACITY is linear in n. This is the only reason the cluster exists.
  * SPEED in plain pipeline (batch=1 relay) is FLAT in n. Only one stage computes
    at a time, so aggregate bandwidth is one machine's bandwidth no matter how
    many you wire up. Adding a machine splits the weights, not the work.
  * SPEED only grows with n under *pipelined speculative decoding*: stages then
    work on different draft chunks concurrently, so tok/s ~ k * n * B / W.
    k = accepted tokens per round, and it is the real bottleneck (n-gram drafts
    measured 0.012 acceptance in Shard -- useless; a real draft model is required).

Roofline, not measurement: per-layer time = layer bytes / usable bandwidth.
Stated assumptions are printed with the output, because they are the whole game.

Run:
    python -m cluster.econ --model 200B --nodes 5 8 16 --k 3 --hop 0.3
"""
import argparse

from .footprint import MODELS


def pick(name):
    for m in MODELS:
        if name.lower() in m.name.lower():
            return m
    raise SystemExit(f"unknown model {name}; try: " +
                     ", ".join(m.name.split()[0] for m in MODELS))


def split(n_layers, n):
    base, rem = divmod(n_layers, n)
    out, lo = [], 0
    for i in range(n):
        hi = lo + base + (1 if i < rem else 0)
        out.append((lo, hi))
        lo = hi
    return out


def node_bytes(m, lo, hi, ctx, with_head):
    w = m.weight_bytes(hi - lo)
    kv = m.kv_bytes_per_token(hi - lo) * ctx
    h = m.head_bytes() if with_head else 0
    return w + kv + h


def run(a):
    m = pick(a.model)
    W = m.weight_bytes(m.n_layers) + m.head_bytes()
    usable = (a.ram - a.reserve) * 1e9
    print(f"模型 {m.name}")
    print(f"  權重 {W/1e9:.1f} GB（{m.bits:g}-bit）+ KV {m.kv_bytes_per_token(m.n_layers)/1e6:.2f} MB/token"
          f" × ctx {a.ctx}")
    print(f"  每台可用 {usable/1e9:.0f} GB（{a.ram}GB 留 {a.reserve}GB 給系統）"
          f"，單機 {'裝得下' if W <= usable else '裝不下 → 這台不存在，必須切'}")
    print(f"  假設：每台可用頻寬 {a.bandwidth} GB/s（M4 mini 120 / Studio 819），"
          f"接力每跳 {a.hop} ms，每段固定 {a.fixed} ms，投機 k={a.k}")

    print("\n n | 每段層 | 最重那台 |  裝得下 | 接力 tok/s | 投機 tok/s | 每 token 通訊")
    print("---+--------+----------+--------+------------+------------+-----------")
    for n in a.nodes:
        b = split(m.n_layers, n)
        peak = max(node_bytes(m, lo, hi, a.ctx, i == 0) for i, (lo, hi) in enumerate(b))
        fit = peak <= usable
        max_layers = max(hi - lo for lo, hi in b)
        # relay: every stage runs in turn, so the weight reads add up
        t_stage = a.fixed + (W / n) / (a.bandwidth * 1e9) * 1e3
        relay_ms = n * t_stage + n * a.hop
        relay = 1e3 / relay_ms
        # pipelined spec: stages overlap; the token rate is set by the slowest stage
        spec_ms = a.fixed + (max_layers * m.weight_bytes(1) / (a.bandwidth * 1e9)) * 1e3
        spec = a.k * 1e3 / (spec_ms + a.hop)
        print(f"{n:2d} | {max_layers:6d} | {peak/1e9:7.1f}G | {'OK' if fit else '  NO':>6} "
              f"| {relay:10.2f} | {spec:10.2f} | {n * a.hop:8.1f}ms")

    if a.compare:
        bw = a.compare_bw
        relay_ms = (W / (bw * 1e9)) * 1e3
        relay = 1e3 / relay_ms
        spec = a.k * 1e3 / relay_ms     # one machine: no pipeline, k drafts per pass
        print(f"\n對照組：一台 {a.compare}（{bw} GB/s，{a.compare_price}）")
        print(f"  容量 {'裝得下' if W <= a.compare_usable * 1e9 else '裝不下'}"
              f" | 接力 {relay:.2f} tok/s | 投機(k={a.k}) {spec:.2f} tok/s"
              f" | 價格 {a.compare_price}")

    print("\n怎麼讀這張表：")
    print("  - 接力欄幾乎與 n 無關（權重讀取時間相加，只是分散到各台輪流做）——"
          "所以『多接一台會更快』在接力模式下是錯的，多接一台只換到容量。")
    print("  - 投機欄才會隨 n 長，因為各段同時在做不同的草稿塊；代價是 k 必須 >1，"
          "而 k 取決於草稿品質，不是接得越多越好。")
    print("  - 最重那台決定可行性：node 0 額外背 embed + lm_head，對稱切分會讓它先爆。")


def main(argv=None):
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="200B")
    ap.add_argument("--nodes", type=int, nargs="+", default=[5, 6, 8, 12, 16])
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--ram", type=float, default=32)
    ap.add_argument("--reserve", type=float, default=8)
    ap.add_argument("--bandwidth", type=float, default=120)
    ap.add_argument("--hop", type=float, default=0.3, help="ms per hop (TB4 TCP estimate)")
    ap.add_argument("--fixed", type=float, default=0.26)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--compare", default="Mac Studio 512GB")
    ap.add_argument("--compare-bw", type=float, default=819)
    ap.add_argument("--compare-usable", type=float, default=512)
    ap.add_argument("--compare-price", default="¥112,899 翻新")
    ap.add_argument("--max-layers-ref", type=int, default=80)
    run(ap.parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
