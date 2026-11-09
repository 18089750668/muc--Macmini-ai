"""Fault injection: kill a stage mid-generation and check what the head does.

The bad outcome is not "it fails" -- it is "it hangs forever and looks like it is
thinking". This asserts the head dies fast with a reason you can act on.
"""
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-arm/bin/python")
TAIL_PORT = 5701
HEAD_PORT = 5700


def find_pid(port):
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def main():
    cmd = [PY, "-m", "cluster.main", "run", "--layers", "0:4",
           "--tail", f"127.0.0.1:{TAIL_PORT}", "--head-port", str(HEAD_PORT),
           "--max-tokens", "6000", "--repeat", "1"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)

    print("等待鏈路起來（JIT 較慢）...", flush=True)
    time.sleep(5)
    pid = find_pid(TAIL_PORT)
    if not pid:
        proc.kill()
        print("找不到 stage 進程，測試環境不對")
        return 1
    print(f"殺掉 stage pid={pid}（模擬節點崩潰/掉線）", flush=True)
    os.kill(pid, signal.SIGKILL)

    t0 = time.time()
    try:
        out = proc.communicate(timeout=40)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        out = proc.communicate()[0]
        print(f"\n[失敗] head 在 40s 內沒有結束——這是最壞情況：看起來像在思考，其實永遠不會回來")
        print(out[-2000:])
        return 1
    dt = time.time() - t0
    print(f"head 在 {dt:.1f}s 內結束，exit={proc.returncode}")
    hit = [ln for ln in out.splitlines()
           if "unresponsive" in ln or "closed the connection" in ln or "RuntimeError" in ln]
    print("判定訊息:")
    for ln in hit[-4:]:
        print("  " + ln.strip())
    ok = bool(hit) and dt < 40
    print("\n[通過] 掉線被明確檢測到，且沒有無限等待" if ok else "\n[失敗] 沒有給出可據以行動的錯誤")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
