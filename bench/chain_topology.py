"""鏈式佈線（每台只接上下家）vs 直連回報：正確性與速度對照。

為什麼要有這支腳本：TB4 每一條線都是「點對點」的獨立 L2 網段，
所以 4 台接成 A—B—C—D（3 條線）時，D 根本沒有到 A 的路由。
本來的實作讓最後一台直接 `--reply <head>`，那只有在同一個 LAN / loopback
上成立。真實鏈路下，回程必須由前一級一級轉發（`--chain`），這支腳本量
「轉發這條回程路徑值多少錢」。

跑法：
    PYTHONPATH=. .venv-arm/bin/python bench/chain_topology.py --nodes 2 3 4
"""
import argparse
import hashlib
import socket
import time

from cluster.config import ToyConfig
from cluster.coordinator import RingHead, generate
from cluster.main import Chain
from cluster.metrics import Metrics


def bounds_of(cfg, nodes):
    n, k = cfg.n_layers, None
    k = n // nodes
    out, lo = [], 0
    for i in range(nodes):
        hi = lo + k if i < nodes - 1 else n
        out.append((lo, hi))
        lo = hi
    return out


def build(cfg, nodes, stage_port, head_port, chain_mode):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", head_port))
    srv.listen(1)
    bounds = bounds_of(cfg, nodes)
    ports = [stage_port + i * 10 for i in range(nodes - 1)]
    ck = Chain(cfg, bounds, ports, head_port, 0, srv, chain=chain_mode)
    ck.start()
    lo, hi = bounds[0]
    head = RingHead(cfg, lo, hi, ck.out_sock, ck.in_sock)
    return ck, head


def run(cfg, nodes, prompt, n_new, stage_port, head_port, chain_mode):
    ck, head = build(cfg, nodes, stage_port, head_port, chain_mode)
    m = Metrics()
    t0 = time.perf_counter()
    out = generate(head, prompt, n_new, mode="greedy", metrics=m)
    wall = time.perf_counter() - t0
    ids = [int(x) for x in out]
    s = m.summary(wall, n_new)
    ck.stop()
    return ids, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--stage-port", type=int, default=7301)
    a = ap.parse_args()
    cfg = ToyConfig()

    # 單機基準（同一顆隨機權重模型，1 台全層）
    from cluster.coordinator import LocalBackend
    be = LocalBackend(cfg)
    prompt = [int(x) for x in range(1000, 1000 + a.prompt_len)]
    m = Metrics()
    t0 = time.perf_counter()
    base = generate(be, prompt, a.tokens, mode="greedy", metrics=m)
    wall = time.perf_counter() - t0
    base_ids = [int(x) for x in base]
    h = hashlib.blake2b(bytes(str(base_ids), "utf8"), digest_size=8).hexdigest()
    print(f"1 台基準：{a.tokens / wall:.2f} tok/s  hash={h}")

    print(f"\n{'台數':<6}{'直連回報 tok/s':>16}{'鏈式轉發 tok/s':>16}"
          f"{'鏈式代價':>12}{'鏈/直一致':>12}{'對單機':>8}")
    for n in a.nodes:
        sp = a.stage_port
        ids_d, sd = run(cfg, n, prompt, a.tokens, sp, 0, False)
        ids_c, sc = run(cfg, n, prompt, a.tokens, sp + 100, sp + 140, True)
        assert ids_d == ids_c, f"{n} 台：鏈式與直連的輸出不同"
        # 單機基準只當參考：玩具模型的 float 結合順序不同，長序列（32 token）
        # 允許漂，但不能漂到跟「直連」不一樣 —— 那才是拓撲/接線的問題。
        same_base = "同" if ids_c == base_ids else "漂"
        cost = sc["tok_s"] / sd["tok_s"]
        print(f"{n:<6}{sd['tok_s']:>16.2f}{sc['tok_s']:>16.2f}"
              f"{cost:>11.2f}x{'一致':>8}{same_base:>6}")
        print(f"       直連 hop={sd['hop_ms']:.2f}ms tau={sd['tavg_ms']:.2f}ms | "
              f"鏈式 hop={sc['hop_ms']:.2f}ms tau={sc['tavg_ms']:.2f}ms")


if __name__ == "__main__":
    main()
