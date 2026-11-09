"""批次解碼（batching）：把 batch=1 的 launch 開銷攤到 B 條序列上。

要驗的假設只有一句：**每步的成本幾乎不隨 B 漲**，所以 tok/s 應該近似隨 B 線性長。
如果這條成立，4 台從「能跑」變成「跑得動」就有了第一根槓桿；如果不成立，那
瓶頸就不是 launch，而是別的（帶寬 / attention），得換方向。

兩個檢查：
  1. 正確性：B 條序列一起跑，每一條的輸出必須跟「它自己單獨跑」逐 token 相同。
  2. 吞吐：B = 1/2/4/8 的每步延遲與 tok/s，單機與 2 台各一組。

    PYTHONPATH=. .venv-arm/bin/python bench/batch_decode.py
"""
import argparse
import socket
import time

from tokenizers import Tokenizer

from cluster import protocol as P
from cluster.coordinator import LocalBackend, RingHead
from cluster.main import Chain
from cluster.realmodel import RealConfig

PATH = "models/qwen2.5-0.5b"

TEXTS = [
    "The capital of France is",
    "In the beginning, the universe was",
    "Machine learning is a field of",
    "Once upon a time, there was a",
]


def pad_rows(tok, texts):
    """等長化：短的補 eos。第一版要求等長（KV 共用同一個位置索引）。"""
    rows = [tok.encode(t).ids for t in texts]
    n = max(len(r) for r in rows)
    pad = tok.token_to_id("<|endoftext|>")
    return [r + [pad] * (n - len(r)) for r in rows], n


def decode_rows(be, rows, n_new, label=None):
    """等長 rows 一起 prefill 再一起 decode；回傳 (每條 token 序列, decode 秒數)。"""
    B, L = len(rows), len(rows[0])
    toks, _, _, _ = be.forward_rows(rows, 0, kind=P.PREFILL)
    out = [[t] for t in toks]
    pos = L
    t0 = time.perf_counter()
    for _ in range(n_new - 1):
        nxt, _, _, _ = be.forward_rows([[o[-1]] for o in out], pos, kind=P.VERIFY)
        for i, t in enumerate(nxt):
            out[i].append(t)
        pos += 1
    dt = time.perf_counter() - t0
    if label:
        print(f"  {label} B={B:<2} {dt/(n_new-1)*1e3:6.1f}ms/step  "
              f"{B*(n_new-1)/dt:6.2f} tok/s")
    return out, dt


def build_ring(bounds, ports, head_port, cfg):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", head_port))
    srv.listen(1)
    chain = Chain(cfg, bounds, ports, head_port, 0, srv, model_dir=PATH, chain=True)
    chain.start()
    head = RingHead(cfg, bounds[0][0], bounds[0][1],
                    chain.out_sock, chain.in_sock, model_dir=PATH)
    return chain, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--batch", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--stage-port", type=int, default=7701)
    ap.add_argument("--head-port", type=int, default=7711)
    args = ap.parse_args()

    tok = Tokenizer.from_file(f"{PATH}/tokenizer.json")
    cfg = RealConfig(PATH)
    rows, L = pad_rows(tok, TEXTS)
    print(f"Qwen2.5-0.5B，{args.tokens} 個新 token，greedy；"
          f"{len(rows)} 條 prompt 等長化到 {L} tokens\n")

    # ---- 單機：batch 是否會把不同列混在一起 -------------------------------
    print("[單機 24 層] 同一批一起跑 vs 每條單獨跑")
    local = LocalBackend(cfg, model_dir=PATH)
    batch_out, dt = decode_rows(local, rows, args.tokens, "全 24 層")
    ok = True
    for i in range(len(rows)):
        solo = LocalBackend(cfg, model_dir=PATH)
        ref, _ = decode_rows(solo, [rows[i]], args.tokens)
        same = ref[0] == batch_out[i]
        ok = ok and same
        mark = "一致" if same else "不一致 ✗"
        print(f"  row {i}: {mark}  {batch_out[i][:8]}")
    print(f"  結果：{'batch 沒有污染任何一列' if ok else 'batch 污染了某一列'}"
          f"（{dt/(args.tokens-1)*1e3:.1f}ms/step、{len(rows)*(args.tokens-1)/dt:.2f} tok/s）")

    # ---- 2 台：同樣的 batch 走真鏈路 ------------------------------------
    print(f"\n[2 台 ring 0:12 | 12:24] stage :{args.stage_port} head :{args.head_port}")
    chain, head = build_ring([(0, 12), (12, 24)], [args.stage_port], args.head_port, cfg)
    try:
        batch_out2, dt2 = decode_rows(head, rows, args.tokens, "2 台")
        print(f"  2 台 vs 單機逐 token 相同：{[a == b for a, b in zip(batch_out, batch_out2)]}")

        ok2 = True
        for i in range(len(rows)):
            head.reset()                       # 清 head 與所有 stage 的 KV
            solo, _ = decode_rows(head, [rows[i]], args.tokens)
            if solo[0] != batch_out2[i]:
                ok2 = False
                print(f"  row {i} 不一致 ✗ batch={batch_out2[i][:6]} solo={solo[0][:6]}")
        print(f"  2 台 batch 正確性：{'全部一致' if ok2 else '有列不一致 ✗'}")
    finally:
        chain.stop(abort=True)

    # ---- 吞吐掃描：B 對每步延遲的影響 -----------------------------------
    print(f"\n[吞吐] 每組重起一條乾淨鏈")
    print(f"{'B':>3}{'1 台 ms/step':>16}{'1 台 tok/s':>13}"
          f"{'2 台 ms/step':>16}{'2 台 tok/s':>13}")
    for i, B in enumerate(args.batch):
        sub = rows[:B]
        one = LocalBackend(cfg, model_dir=PATH)
        _, dt1 = decode_rows(one, sub, args.tokens)
        # 每個 B 換一組端口：反覆在同一端口重建，會踩到 macOS 的 TIME_WAIT
        # （_check_ports 的 bind 探測刻意不帶 SO_REUSEADDR，所以它會正確地擋）
        chain, head = build_ring([(0, 12), (12, 24)],
                                 [args.stage_port + i * 20],
                                 args.head_port + i * 20, cfg)
        try:
            _, dt2 = decode_rows(head, sub, args.tokens)
        finally:
            chain.stop(abort=True)
        print(f"{B:>3}{dt1/(args.tokens-1)*1e3:16.1f}"
              f"{B*(args.tokens-1)/dt1:13.2f}"
              f"{dt2/(args.tokens-1)*1e3:16.1f}"
              f"{B*(args.tokens-1)/dt2:13.2f}")

    print("\n讀數：每步延遲若不隨 B 漲，tok/s 就該近似隨 B 線性長。")
    print("      單機與 2 台都受益，2 台只是每步多付一次 hop。")


if __name__ == "__main__":
    main()
