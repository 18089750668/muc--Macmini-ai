"""Does the split actually fit? Layer-by-layer memory accounting.

The planner so far only minimises TIME. The thing that kills a real cluster is
MEMORY, and it is asymmetric: weights and KV both live on the node that holds
those layers, so a node with more layers pays twice. This script answers, for a
given model and node budget:

  - how much weight memory each node carries
  - how much KV it must hold at a given context (this is the number that grows)
  - the longest context it can serve before hitting the node's usable RAM
"""
import argparse
import math
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    bits: float = 4.0          # weight bits per parameter
    kv_bits: float = 16.0      # KV cache precision

    @property
    def kv_hidden(self):
        return self.n_kv_heads * self.head_dim

    def params_per_layer(self):
        h, kvh, ffn = self.hidden, self.kv_hidden, self.ffn
        return h * (self.n_heads * self.head_dim) + 2 * h * kvh \
            + (self.n_heads * self.head_dim) * h + 3 * h * ffn

    def total_params(self):
        return self.params_per_layer() * self.n_layers + 2 * self.vocab * self.hidden

    def weight_bytes(self, layers):
        return self.params_per_layer() * layers * self.bits / 8

    def kv_bytes_per_token(self, layers):
        return 2 * self.kv_hidden * layers * self.kv_bits / 8

    def head_bytes(self):
        return 2 * self.vocab * self.hidden * self.bits / 8


MODELS = [
    # 8B 级 GQA，fp16 KV
    Model("8B  (28L h4096 kv8 ffn12288)", 28, 4096, 32, 8, 128, 12288, 152064, 4.0, 16.0),
    # 70B 级
    Model("70B (80L h8192 kv8 ffn28672)", 80, 8192, 64, 8, 128, 28672, 152064, 4.0, 16.0),
    # 200B 级 MoE-ish 稠密等价
    Model("200B(80L h12288 kv8 ffn49152)", 80, 12288, 96, 8, 128, 49152, 152064, 4.0, 16.0),
]


def split(n_layers, nodes):
    base, rem = divmod(n_layers, nodes)
    return [base + (1 if i < rem else 0) for i in range(nodes)]


def report(model, nodes, ctx, ram_gb=32.0, reserve_gb=8.0):
    usable = (ram_gb - reserve_gb) * 1e9
    shares = split(model.n_layers, nodes)
    w = [model.weight_bytes(s) for s in shares]
    kv1 = [model.kv_bytes_per_token(s) for s in shares]
    kv = [k * ctx for k in kv1]
    peak = max(wi + ki for wi, ki in zip(w, kv))
    # per-node head cost (embed + lm_head) sits on node 0 only
    extra = [model.head_bytes() if i == 0 else 0.0 for i in range(len(shares))]
    peak = max(wi + ki + ei for wi, ki, ei in zip(w, kv, extra))
    fits = peak <= usable
    max_ctx = None
    if kv1:
        caps = [(usable - w[i] - extra[i]) / kv1[i] for i in range(len(shares))]
        max_ctx = min(caps) if caps else None
        if max_ctx is not None and max_ctx < 0:
            max_ctx = None          # 權重本身就放不下，不談上下文
    print(f"\n{model.name}  總參數 {model.total_params() / 1e9:.1f}B  "
          f"4bit 權重 {model.total_params() * 0.5 / 1e9:.0f}GB")
    print(f"  {nodes} 節點 × {ram_gb:.0f}GB（可用 {usable / 1e9:.0f}GB）  ctx={ctx}")
    print(f"  {'#':>2} {'layers':>7} {'weights':>9} {'head':>7} {'KV':>9} {'sum':>9}")
    for i, s in enumerate(shares):
        tot = w[i] + kv[i] + extra[i]
        flag = "  <-- 爆" if tot > usable else ""
        print(f"  {i:>2} {s:>7} {w[i] / 1e9:>8.1f}G {extra[i] / 1e9:>6.1f}G "
              f"{kv[i] / 1e9:>8.1f}G {tot / 1e9:>8.1f}G{flag}")
    print(f"  峰值 {peak / 1e9:.1f}G  {'OK' if fits else '不通過'}"
          + (f"   最長可跑上下文 ≈ {max_ctx / 1e3:.0f}K token" if max_ctx else ""))
    return dict(model=model.name, nodes=nodes, peak=peak / 1e9, fits=fits,
                max_ctx=(max_ctx / 1e3 if max_ctx else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--ram", type=float, default=32.0)
    ap.add_argument("--reserve", type=float, default=8.0,
                    help="macOS + 進程 + 激活值等固定佔用 (GB)")
    a = ap.parse_args()
    if a.summary:
        print(f"{'模型':<34} {'節點':>4} {'峰值GB':>8} {'最長ctx':>10}  判定")
        for m in MODELS:
            for n in (2, 4, 8, 16):
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    r = report(m, n, a.ctx, a.ram, a.reserve)
                print(f"{r['model']:<34} {r['nodes']:>4} {r['peak']:>8.1f} "
                      f"{r['max_ctx']:>9.0f}K   {'OK' if r['fits'] else '不通過'}")
        return
    for m in MODELS:
        for n in (4, 8, 16):
            report(m, n, a.ctx, a.ram, a.reserve)
    print("\n注意：權重放得下 ≠ 能跑長上下文。KV 只掛在持有該層的節點上，")
    print("所以「層多的那個節點」同時背最重的權重和最多的 KV——分配必須按記憶體再校驗一次。")


if __name__ == "__main__":
    main()
