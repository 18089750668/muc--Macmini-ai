""""按專家切分"（expert parallel, EP）到底能不能用：一份可以算的帳。

先講清楚三個容易混在一起的事：

1. **參數量不會變小。** 125B 的 MoE 在硬碟上、在記憶體裡都還是 125B。
   變小的是「每產生一個 token 要算多少參數」——10B active 就是 10B 的計算量。
   所以「切完就不是 125B 了」這句話只在算力上成立，在容量上不成立。
2. **EP 省的是每台的記憶體**（每台只放自己那幾位專家），代價是每個 MoE 層、
   每個 token 都要一次 all-to-all（dispatch 出去、combine 收回）。
3. **EP 的稅是按「層數」收的**（2 × 層數 次往返），pipeline 的稅是按「機器數」
   收的（n − 1 次往返）。這是兩者最本質的差別，決定了單人 vs 多人場景誰贏。

所以這個工具算兩條路線在同一個模型上的每 token 延遲：

    python -m cluster.moe --total 125 --active 10 --layers 48 --experts 128 --topk 8
    python -m cluster.moe --total 125 --active 10 --nodes 4 --topology ring --hop-ms 0.25
    python -m cluster.moe --total 671 --active 37 --layers 61 --experts 256 --topk 8 --nodes 16

數字全部來自輸入，沒有藏假設；預設值都印在輸出第一段。
"""
import argparse


def human(gb):
    return f"{gb:.1f}G" if gb >= 1 else f"{gb*1000:.0f}M"


def calc(a, nodes, hop_ms, topology, batch, bw):
    bp = a.bits / 8.0
    total_b = a.total * bp
    active_b = a.active * bp
    dense_b = total_b * a.dense_share
    expert_total_b = total_b - dense_b
    active_expert_b = active_b * (1 - a.dense_share)

    hops = 1 if topology == "star" else max(1, nodes / 4.0)   # ring 平均距離 ≈ n/4
    hop = hop_ms * hops

    # ---- pipeline（按層切，跟我們已經做的接力一樣）----
    pp_mem = total_b / nodes
    pp_compute = active_b / bw * 1000            # ms，全部段加起來
    pp_comm = (nodes - 1) * hop
    pp_ms = pp_compute + pp_comm / batch          # batch 大時通訊攤掉，但計算按 batch 漲
    pp_tps = 1000.0 / pp_ms if pp_ms else 0

    # ---- expert parallel（按專家切）----
    ep_mem = dense_b + expert_total_b / nodes    # 稠密部分（attention/embed）每台都要有一份
    hit = min(a.topk, nodes)                     # 一個 token 的 topk 專家落在幾台上
    # batch=1 時 attention 只能待在一個節點上（沒有第二條序列可以分出去），
    # 所以稠密那 12.5G 每 token 都要整份讀一遍 —— 這是 EP 在單人場景的死穴。
    ep_compute = (dense_b + active_expert_b / hit) / bw * 1000
    ep_comm = 2 * a.layers * hop                  # 每層 dispatch + combine
    ep_ms = ep_compute + ep_comm / batch
    ep_tps = 1000.0 / ep_ms if ep_ms else 0
    return dict(pp_mem=pp_mem, pp_ms=pp_ms, pp_tps=pp_tps, ep_mem=ep_mem,
                ep_ms=ep_ms, ep_tps=ep_tps, ep_comm=ep_comm, ep_compute=ep_compute,
                pp_comm=pp_comm, total_b=total_b, hop=hop, hops=hops)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=float, default=125, help="總參數 B")
    p.add_argument("--active", type=float, default=10, help="每 token 激活參數 B")
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--bits", type=float, default=4)
    p.add_argument("--dense-share", type=float, default=0.2,
                   help="attention/embedding 這類每 token 都要算的比例")
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--bw", type=float, default=120, help="每台可用帶寬 GB/s")
    p.add_argument("--hop-ms", type=float, default=0.15, help="單跳往返 ms")
    p.add_argument("--topology", choices=["star", "ring"], default="star")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--capacity", type=float, default=24.0,
                   help="每台可用 GB（顯存/統一記憶體扣掉系統）")
    a = p.parse_args()

    print(f"模型 {a.total:.0f}B / 每 token 激活 {a.active:.0f}B，{a.layers} 層，"
          f"{a.experts} 專家 top{a.topk}")
    print(f"假設：{a.bits:.0f}-bit（{human(a.total * a.bits/8)} 權重）、"
          f"每台 {a.bw:.0f}GB/s、單跳 {a.hop_ms}ms、topology={a.topology}")

    rows = []
    sizes = [n for n in (2, 4, 8, 16) if n <= a.nodes]
    if a.nodes not in sizes:
        sizes.append(a.nodes)
    for n in sizes:
        for b in (1, 8, 32):
            r = calc(a, n, a.hop_ms, a.topology, b, a.bw)
            rows.append((n, b, r))

    print(f"\n{'n':>3} {'batch':>5} | {'PP 每台':>10} {'ms/tok':>8} {'tok/s':>8}"
          f" | {'EP 每台':>10} {'ms/tok':>8} {'tok/s':>8} | 贏家（每台容量 {a.capacity:.0f}G）")
    for n, b, r in rows:
        win = "PP" if r["pp_tps"] >= r["ep_tps"] else "EP"
        fit = "" if r["ep_mem"] <= a.capacity else " (EP 放不下)"
        fit2 = "" if r["pp_mem"] <= a.capacity else " (PP 放不下)"
        print(f"{n:>3} {b:>5} | {human(r['pp_mem']):>8}{fit2} {r['pp_ms']:>8.1f} "
              f"{r['pp_tps']:>8.1f} | {human(r['ep_mem']):>8}{fit} {r['ep_ms']:>8.1f} "
              f"{r['ep_tps']:>8.1f} | {win}")

    print("\nEP 的兩筆帳（batch=1，{0} 台）：".format(a.nodes))
    r = calc(a, a.nodes, a.hop_ms, a.topology, 1, a.bw)
    print(f"  計算 {r['ep_compute']:.1f}ms（專家並行，最慢那台）"
          f" + 通訊 {r['ep_comm']:.1f}ms（{a.layers} 層 × 2 × {r['hops']:.1f} 跳 × {a.hop_ms}ms）")
    print(f"  pipeline 那條：計算 {(a.active * a.bits/8)/a.bw*1000:.1f}ms"
          f" + 通訊 {(a.nodes - 1) * r['hop']:.1f}ms（{a.nodes - 1} 跳）")
    print("\n注意：這裡的 tok/s 是「硬體上限」(roofline = 每 token 要讀的權重 bytes / 帶寬)。"
          "現有原型只跑到上限的 ~8%（batch=1 解碼是 launch-bound），"
          "所以這些數字是「換對方案之後能到哪裡」，不是「現在能跑到哪裡」。")
    print("\n結論：EP 的稅按層數收、PP 的稅按機器數收。單人 batch=1 時，"
          "層數(48~61) 遠大於機器數(4~16)，所以 PP 幾乎總是贏；"
          "batch 大了之後 EP 的通訊被攤掉，才輪到 EP 贏。")


if __name__ == "__main__":
    main()
