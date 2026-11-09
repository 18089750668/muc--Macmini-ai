"""量 k：投機解碼每輪實際被接受幾個 token（這是整條路線唯一真正的瓶頸）。

為什麼量這個：速度 ≈ (tau_chain + hop) / g，g = 每輪接受的 token 數。
g = 1 就是「送 block 個候選、只收 1 個」，純虧；g = 3 才開始賺。
之前真模型上量到 g = 1.00，那是因為草稿是「重複上一個 token」這種佔位實作。

這支 bench 換三個真的草稿來源（見 cluster/draft.py），並**先驗正確性**：
投機的輸出必須跟 greedy 逐 token 相同（驗證只接受對得上的前綴，錯了就是 bug）。

    PYTHONPATH=. .venv-arm/bin/python bench/spec_k.py
    PYTHONPATH=. .venv-arm/bin/python bench/spec_k.py --nodes 2
"""
import argparse
import socket
import time

from tokenizers import Tokenizer

from cluster import protocol as P
from cluster.coordinator import LocalBackend, RingHead, generate
from cluster.draft import (HitStats, LookupDraft, MixedDraft, RepeatDraft,
                           SelfDraft)
from cluster.main import Chain
from cluster.metrics import Metrics
from cluster.realmodel import RealConfig

PATH = "models/qwen2.5-0.5b"

CASES = {
    "散文": "The capital of France is",
    "清單": "The planets in order from the sun are: 1. Mercury 2. Venus 3. Earth 4. Mars 5. Jupiter 6.",
    "程式": ("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\n"
             "def mul(a, b):\n    return a * b\n\ndef div(a, b):\n    return a /"),
    "照抄": ("Repeat the following sentence exactly: the quick brown fox jumps over "
             "the lazy dog and then runs back to the barn. the quick brown fox jumps"),
}


def drafters(cfg, args):
    out = []
    for tag in [t.strip() for t in args.drafters.split(",") if t.strip()]:
        if tag == "repeat":
            out.append((tag, RepeatDraft()))
        elif tag == "lookup":
            out.append((tag, LookupDraft(args.n)))
        elif tag == "mixed":
            out.append((tag, MixedDraft(args.n)))
        elif tag == "self":
            out.append((tag, SelfDraft(cfg, PATH, layers=args.draft_layers)))
        else:
            raise SystemExit(f"unknown drafter {tag}")
    return out


def run(be, ids, n_new, mode, drafter=None, block=4):
    m = Metrics()
    st = HitStats()
    t0 = time.perf_counter()
    out = generate(be, ids, n_new, block=block, mode=mode, metrics=m, drafter=drafter,
                   stats=st)
    wall = time.perf_counter() - t0
    s = m.summary(wall, len(out))
    return out, s, st.summary()


def build_ring(cfg, nodes, stage_port, head_port):
    """N 台接力：層數平均切，stage 埠每台錯開 10。"""
    n = cfg.n_layers
    k = n // nodes
    bounds, lo = [], 0
    for i in range(nodes):
        hi = lo + k if i < nodes - 1 else n
        bounds.append((lo, hi))
        lo = hi
    ports = [stage_port + i * 10 for i in range(nodes)]
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", head_port))
    srv.listen(1)
    chain = Chain(cfg, bounds, ports, head_port, 0, srv, model_dir=PATH)
    chain.start()
    lo, hi = bounds[0]
    head = RingHead(cfg, lo, hi, chain.out_sock, chain.in_sock, model_dir=PATH)
    return chain, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--n", type=int, default=3, help="lookup 的 n-gram 長度")
    ap.add_argument("--drafters", default="repeat,lookup,mixed",
                    help="逗號分隔；self 需要 --draft-layers")
    ap.add_argument("--draft-layers", type=int, default=4)
    ap.add_argument("--stage-port", type=int, default=7761)
    ap.add_argument("--head-port", type=int, default=7771)
    args = ap.parse_args()

    tok = Tokenizer.from_file(f"{PATH}/tokenizer.json")
    cfg = RealConfig(PATH)
    chain = head = None
    if args.nodes >= 2:
        chain, head = build_ring(cfg, args.nodes, args.stage_port, args.head_port)
        be = head
    else:
        be = LocalBackend(cfg, model_dir=PATH)

    print(f"{args.nodes} 台，Qwen2.5-0.5B，{args.tokens} 個新 token，"
          f"block={args.block}，lookup n={args.n}，greedy 為基準")
    tags = [t for t, _ in drafters(cfg, args)]
    print(f"{'case':<6}{'greedy':>10}" + "".join(f"{t:>18}" for t in tags)
          + f"{'一致':>6}")
    try:
        for name, text in CASES.items():
            ids = tok.encode(text).ids
            if chain is not None:
                head.reset()
            g_out, g_s, _ = run(be, ids, args.tokens, "greedy", block=args.block)
            cells, ok = [], True
            for tag, d in drafters(cfg, args):
                if chain is not None:
                    head.reset()
                out, s, st = run(be, ids, args.tokens, "spec", d, block=args.block)
                ok = ok and (out == g_out)
                cells.append(f"{s['tok_s']:5.2f}t/s k={s['g'] - 1:.2f}")
                if not st["rounds"]:
                    cells[-1] += " (無)"
            print(f"{name:<6}{g_s['tok_s']:10.2f}"
                  + "".join(f"{c:>18}" for c in cells)
                  + f"{'一致' if ok else '不一致 ✗':>6}")
        print("\n讀數：g = 每輪接受的 token 數（含 bonus），k = g − 1 = 草稿被接受的個數。")
        print("      k 隨文字類型變化很大：重複結構（程式/照抄）高，自由散文接近 0。")
        print("      單人延遲 ≈ (tau_chain + hop) / g，所以 g 才是單人速度的全部。")
    finally:
        if chain is not None:
            chain.stop(abort=True)


if __name__ == "__main__":
    main()
