"""serve 層的容錯：**掉線之後那條正在跑的請求要續跑，不是報錯。**

鏈路層的重放能力（`RingHead.replay`）已經在 `fault_recover.py` 裡驗過，但它是
「整條鏈重建 + 重放整段前綴」。這裡驗的是服務層：一條請求已經吐了 N 個 token
之後中間那台被 SIGKILL，客戶端應該看到「繼續串流到結束」，而且**整段文字與
沒有掉線時逐字相同**。

做法：同一個 prompt 跑兩次，第一次不掉線當基準，第二次在生成中途殺掉 stage。
    PYTHONPATH=. .venv-arm/bin/python bench/serve_fault.py
"""
import argparse
import json
import socket
import subprocess
import sys
import time

PATH = "models/qwen2.5-0.5b"
PROMPT = ("In the beginning, the universe was very hot and dense, and then it began "
          "to expand. As it expanded it cooled, and the first atoms formed. ")


def wait_port(port, tries=120):
    for _ in range(tries):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def request(port, prompt, max_tokens, box):
    s = socket.create_connection(("127.0.0.1", port), timeout=600)
    f = s.makefile("rwb")
    f.write(json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode() + b"\n")
    f.flush()
    text, done, err = "", None, None
    while True:
        line = f.readline()
        if not line:
            break
        msg = json.loads(line)
        if "delta" in msg:
            text += msg["delta"]
            box["deltas"] += 1
        elif msg.get("done"):
            done = msg
        elif "error" in msg:
            err = msg["error"]
            break
    f.close()
    s.close()
    return text, done, err


def kill_stage(port):
    pids = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                          capture_output=True, text=True).stdout.split()
    for pid in pids:
        subprocess.run(["kill", "-9", pid])
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--after", type=float, default=2.0, help="幾秒後殺 stage")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--stage-port", type=int, default=7811)
    ap.add_argument("--head-port", type=int, default=7911)
    ap.add_argument("--mode", default="spec", choices=["batch", "spec", "auto"])
    args = ap.parse_args()

    log = open("/tmp/serve_fault.log", "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "cluster.serve", "--model-dir", PATH, "--nodes", "2",
         "--stage-port", str(args.stage_port), "--head-port", str(args.head_port),
         "--port", str(args.port), "--mode", args.mode],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_port(args.port):
            print("服務器沒起來，看 /tmp/serve_fault.log")
            return
        time.sleep(1.0)

        print(f"1) 基準：不掉線，{args.tokens} token")
        t0 = time.perf_counter()
        base, done, err = request(args.port, PROMPT, args.tokens, {"deltas": 0})
        print(f"   {time.perf_counter() - t0:.2f}s, {done and done['tokens']} token, "
              f"err={err}")

        print(f"2) 掉線：生成 {args.after}s 後 SIGKILL 中間那台（stage :{args.stage_port}）")
        box = {"deltas": 0}
        t0 = time.perf_counter()
        import threading
        killer = threading.Timer(args.after, lambda: print(
            "   killed pids:", kill_stage(args.stage_port), flush=True))
        killer.start()
        got, done2, err2 = request(args.port, PROMPT, args.tokens, box)
        killer.cancel()
        wall = time.perf_counter() - t0
        print(f"   {wall:.2f}s, {done2 and done2['tokens']} token, err={err2}")
        print(f"   串流段數 {box['deltas']}（掉線後沒有重送 = 客戶端看不到重複）")
        print(f"   與基準逐字相同：{got == base}")
        if got != base:
            for i, (x, y) in enumerate(zip(base, got)):
                if x != y:
                    print(f"   第一個差異在字元 {i}: base={base[i:i+30]!r} got={got[i:i+30]!r}")
                    break
        print("\n服務器日誌尾段：")
        log.close()
        for line in open("/tmp/serve_fault.log", "rb").read().decode().splitlines()[-8:]:
            print("  " + line)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
