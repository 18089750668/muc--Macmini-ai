"""Does recovery actually recover, or does it just fail politely?

Two runs of the SAME 4-node chain: one clean, one with a stage SIGKILLed
mid-generation. The only acceptable result is identical token hashes -- a
"recovery" that continues from a skewed KV would print tokens happily and be
wrong, which is worse than crashing.
"""
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-arm/bin/python")
PROMPT_LEN = 512
MAX_TOKENS = 400


def run_chain(stage_port, head_port, kill_after=None, timeout=180):
    cmd = [PY, "-m", "cluster.main", "run", "--nodes", "4",
           "--stage-port", str(stage_port), "--head-port", str(head_port),
           "--prompt-len", str(PROMPT_LEN), "--max-tokens", str(MAX_TOKENS),
           "--on-loss", os.environ.get("ON_LOSS", "rebuild")]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    if kill_after:
        if not wait_port(stage_port + 1, timeout=30):
            print("  stage 一直沒起來", flush=True)
        time.sleep(kill_after)
        pid = listen_pid(stage_port + 1)          # the middle stage
        if pid:
            os.kill(pid, signal.SIGKILL)
            print(f"  殺掉中間 stage pid={pid} (port {stage_port + 1})", flush=True)
        else:
            print("  找不到中間 stage 進程", flush=True)
    out = proc.communicate(timeout=timeout)[0]
    return out, proc.returncode


def wait_port(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if listen_pid(port):
            return True
        time.sleep(0.3)
    return False


def listen_pid(port):
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def hash_of(out):
    m = re.search(r"tokens=\d+ hash=(\w+)", out)
    return m.group(1) if m else None


def main():
    print("[1/2] 乾淨的 4 台（基準）", flush=True)
    clean, rc1 = run_chain(5960, 5950)
    h1 = hash_of(clean)
    print(f"  hash={h1} exit={rc1}")

    print("[2/2] 同樣的 4 台，跑到一半殺掉中間 stage", flush=True)
    killed, rc2 = run_chain(5970, 5980, kill_after=2.5)
    h2 = hash_of(killed)

    lines = [ln for ln in killed.splitlines()
             if "掉线" in ln or "判定" in ln or "重放" in ln or "重新分配" in ln
             or "recover" in ln or "自我检测" in ln]
    print("  主機日誌：")
    for ln in lines[:8]:
        print("   " + ln.strip())
    print(f"  hash={h2} exit={rc2}")

    ok = h1 is not None and h1 == h2
    print()
    if ok:
        print(f"[通過] 掉線 → 重建 → 重放前綴 → 續跑，{MAX_TOKENS} 個 token 與乾淨跑逐位一致")
    else:
        print(f"[失敗] 恢復後輸出不一致（乾淨={h1} 恢復={h2}）—— 這比崩潰更危險")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
