"""容量階梯：N 台 32GB 到底能裝什麼。

這個檔案只回答一個問題，而且答案必須是算得出來的數，不是形容詞：

    我手上 N 台 32GB 的機器，合起來能跑多大的模型、能開多長的上下文？

它不是「多台疊起來更快」——那是另一本帳（見 econ.py / moe.py）。這裡只算容量：
每台的權重 + 它自己要扛的 KV，必須塞進它的可用記憶體。

三條會讓人算錯的地方，這裡都明確處理：

1. **可用記憶體 ≠ 標稱記憶體。** 32GB 的 Mac 扣掉系統與 Metal 保留，實際能給
   模型的大約 24GB；16GB 的機器大約 12GB。用標稱值算出來的表是騙人的。
2. **KV 掛在「持有那幾層」的那台身上**，跟權重同一本帳。所以層數多的那台付兩次。
3. **KV 隨上下文線性長**，而權重不長。短上下文時瓶頸是權重，長上下文時是 KV——
   70B 在 32K 上下文的 KV 就有 10GB 量級，比它自己一層的權重還重。

Run:
    python -m cluster.capacity                 # 預設：4/8/16 台 32GB + Studio 對照
    python -m cluster.capacity --ctx 32768
"""
import argparse

from .footprint import Model


SPECS = [
    # name,          layers, hidden, heads, kv_heads, head_dim, ffn,    vocab
    ("8B",            28, 4096, 32, 8, 128, 12288, 152064),
    ("14B",           48, 5120, 40, 8, 128, 13824, 152064),
    ("32B",           64, 5120, 40, 8, 128, 27648, 152064),
    ("70B",           80, 8192, 64, 8, 128, 28672, 152064),
    ("120B(dense)",  100, 9216, 72, 8, 128, 36864, 152064),
    ("200B(dense)",   80, 12288, 96, 8, 128, 49152, 152064),
]

# 名稱, 台數, 標稱 GB, 可用 bytes（實測經驗：扣掉系統與 Metal 保留）
GB = 1e9
RIGS = [
    ("1× 32GB",  1,  32, 24 * GB),
    ("2× 32GB",  2,  32, 24 * GB),
    ("4× 32GB",  4,  32, 24 * GB),
    ("6× 32GB",  6,  32, 24 * GB),
    ("8× 32GB",  8,  32, 24 * GB),
    ("16× 32GB", 16, 32, 24 * GB),
    ("1× 512GB Studio", 1, 512, 480 * GB),
]


def build(spec, bits, kv_bits=16.0):
    name, layers, hidden, heads, kv_heads, head_dim, ffn, vocab = spec
    return Model(name, layers, hidden, heads, kv_heads, head_dim, ffn, vocab,
                 bits, kv_bits)


def plan(m, n, ctx):
    """回傳 (每台最大 bytes, 最少台數是否裝得下, 每台層數)"""
    per = m.weight_bytes(m.n_layers) / n + m.kv_bytes_per_token(m.n_layers) * ctx / n
    # 不對稱：node 0 額外背 embed + lm_head
    per_n0 = per + m.head_bytes()
    return max(per, per_n0), per


def min_nodes(m, ctx, capacity):
    for n in range(1, m.n_layers + 1):
        need, _ = plan(m, n, ctx)
        if need <= capacity:
            return n
    return None


def max_ctx(m, n, capacity):
    """二分找最大上下文（層數切 n 份，最重那台決定）"""
    if min_nodes(m, 1, capacity) == 1:
        lo, hi = 0, 1 << 20
    else:
        if min_nodes(m, n, capacity) is None:
            return 0
        lo, hi = 0, 1 << 20
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if plan(m, n, mid)[0] <= capacity:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--ctx-sweep", type=int, nargs="*", default=[4096, 8192, 32768])
    p.add_argument("--bits", type=float, nargs="*", default=[4.0, 8.0])
    a = p.parse_args()

    print(f"可用記憶體：32GB 機器 → 24GB（扣系統 + Metal 保留）；512GB Studio → 480GB")
    print(f"KV 精度 fp16；每台的帳 = 自己那幾層的權重 + 自己那幾層的 KV，"
          f"node 0 額外背 embed + lm_head\n")

    for bits in a.bits:
        print(f"=== 權重 {bits:.0f}-bit ===")
        emb = build(SPECS[-1], bits).head_bytes() / 2 / GB      # 單份 embed/lm_head
        print(f"（提醒：載入 embed/lm_head 時會多一份暫存，200B 級約 {2*emb:.1f}G；"
              f"頭尾那台要多留這個量，見 bench/verify_capacity.py）")
        head = f"{'機型':<18}{'模型':<14}" + "".join(f"{c//1024:>6}K ctx     " for c in a.ctx_sweep)
        print(head)
        for rig_name, n, nominal, cap in RIGS:
            for spec in SPECS:
                m = build(spec, bits)
                cells = []
                for ctx in a.ctx_sweep:
                    need, _ = plan(m, n, ctx)
                    if need <= cap:
                        cells.append(f"OK {need/GB:5.1f}G")
                    else:
                        mn = min_nodes(m, ctx, cap)
                        cells.append(f"需{mn}台" if mn and mn <= 64 else "  ✗   ")
                print(f"{rig_name:<18}{m.name:<14}" + "".join(f"{c:>14}" for c in cells))
            print()

        # 最小台數與最大上下文，針對 4× 與 8×
        for rig_name, n, nominal, cap in RIGS:
            if n not in (4, 8):
                continue
            for spec in SPECS:
                m = build(spec, bits)
                mn = min_nodes(m, a.ctx, cap)
                mc = max_ctx(m, n, cap)
                if mn is None or mn > 64:
                    print(f"{rig_name} 跑 {m.name}: 一台一層都放不下")
                else:
                    print(f"{rig_name} 跑 {m.name}: {a.ctx//1024}K 上下文最少需 {mn} 台；"
                          f"這 {n} 台最多能開 {mc//1024}K 上下文")
            print()


if __name__ == "__main__":
    main()
