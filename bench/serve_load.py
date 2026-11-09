"""對 serve.py 加負載：驗「多條請求一起跑」不會污染輸出，且吞吐真的隨併發長。

檢查兩件事：
  1. 正確性：每條 prompt 先單獨跑一次記下文字，再讓所有 prompt 同時跑，
     逐字相同才算過（左補齊 + per-row 遮罩最容易在這裡出錯）。
  2. 吞吐：併發 1 / 2 / 4 / 8 的總 tok/s 與單條延遲。

    PYTHONPATH=. .venv-arm/bin/python bench/serve_load.py --nodes 2
"""
import argparse
import json
import socket
import statistics
import subprocess
import sys
import threading
import time

PATH = "models/qwen2.5-0.5b"

PROMPTS = [
    "The capital of France is",
    "In the beginning, the universe was very hot and dense, and then it began to expand",
    "Machine learning is a field of computer science that gives computers the ability to learn without being explicitly programmed. It is",
    "Once upon a time, there was a little girl who lived in a village near the edge of a great forest. Every morning she",
]


def one_request(port, prompt, max_tokens, result, idx):
    t0 = time.perf_counter()
    s = socket.create_connection(("127.0.0.1", port), timeout=300)
    f = s.makefile("rwb")
    f.write(json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode() + b"\n")
    f.flush()
    text, n = "", 0
    while True:
        line = f.readline()
        if not line:
            break
        msg = json.loads(line)
        if msg.get("error"):
            result[idx] = {"error": msg["error"], "sec": time.perf_counter() - t0}
            break
        if msg.get("done"):
            text = msg.get("text", text)
            n = msg.get("tokens", n)
            break
        text += msg.get("delta", "")
    f.close()
    s.close()
    result[idx] = {"text": text, "n": n, "sec": time.perf_counter() - t0}


def run_group(port, prompts, max_tokens):
    res = [None] * len(prompts)
    ts = [threading.Thread(target=one_request, args=(port, p, max_tokens, res, i))
          for i, p in enumerate(prompts)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    return res, wall


def wait_port(port, tries=120):
    for _ in range(tries):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=1)
            s.close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--clients", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--stage-port", type=int, default=7801)
    ap.add_argument("--head-port", type=int, default=7901)
    ap.add_argument("--window-ms", type=float, default=15.0)
    ap.add_argument("--mode", default="batch", choices=["batch", "spec", "auto"],
                    help="spec 只適合單條（多條會排隊），所以併發掃描請用 batch")
    args = ap.parse_args()

    log = open("/tmp/serve_load.log", "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "cluster.serve", "--model-dir", PATH,
         "--nodes", str(args.nodes), "--stage-port", str(args.stage_port),
         "--head-port", str(args.head_port), "--port", str(args.port),
         "--max-batch", "8", "--window-ms", str(args.window_ms),
         "--mode", args.mode],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_port(args.port):
            print("服務器沒起來，看 /tmp/serve_load.log")
            return
        time.sleep(1.0)

        # 1) 基準：每條單獨跑
        base, _ = run_group(args.port, [PROMPTS[0]], args.tokens)
        print(f"[{args.nodes} 台, mode={args.mode}] 單條基準：{base[0]['sec']:.2f}s / "
              f"{base[0]['n']} tokens = {base[0]['n']/base[0]['sec']:.2f} tok/s")

        seq = {}
        for i, p in enumerate(PROMPTS):
            r, _ = run_group(args.port, [p], args.tokens)
            seq[i] = r[0]["text"]

        print(f"\n[{args.nodes} 台] 併發掃描（max_batch=8, window={args.window_ms:.0f}ms）")
        print(f"{'併發':>4}{'總 tok/s':>12}{'單條延遲中位':>16}{'一致':>8}")
        for n in args.clients:
            ps = PROMPTS[:n]
            res, wall = run_group(args.port, ps, args.tokens)
            toks = sum(r["n"] for r in res if r and "n" in r)
            lat = statistics.median([r["sec"] for r in res if r and "sec" in r])
            same = all(res[i]["text"] == seq[i] for i in range(n) if res[i])
            print(f"{n:>4}{toks/wall:12.2f}{lat:16.2f}{'一致' if same else '不一致 ✗':>8}")
            if not same:
                for i in range(n):
                    if res[i] and res[i]["text"] != seq[i]:
                        print(f"     row {i} 併發={res[i]['text'][:40]!r} "
                              f"單獨={seq[i][:40]!r}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
