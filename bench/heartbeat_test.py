"""心跳與自適應門檻：15 秒固定超時到底錯在哪。

三個要分開的情況：
  1. 正常：往返 4ms 的鏈，等 15 秒才算異常 —— 寬容 3750 倍。
  2. 卡住（SIGSTOP）：socket 開著、機器不算。這是遠程最常見的「好像死了」。
  3. 死掉（SIGKILL）：連線直接斷，EOF 立刻到，本來就不必等 15 秒。

這裡量 2 和 3 各自要多久才被主機發現，並確認 1 的基線有被量到。

    PYTHONPATH=. .venv-arm/bin/python bench/heartbeat_test.py
"""
import socket
import time

from cluster import protocol as P
from cluster.config import ToyConfig
from cluster.coordinator import RingHead, StageLost
from cluster.main import Chain


def build(port):
    cfg = ToyConfig()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    bounds = [(0, 4), (4, 8)]
    chain = Chain(cfg, bounds, ports=[port + 1], head_port=port, spin_us=0,
                  srv=srv)
    chain.start()
    head = RingHead(cfg, bounds[0][0], bounds[0][1], chain.out_sock,
                    chain.in_sock, timeout_s=15.0)
    return chain, head


def main():
    cfg = ToyConfig()
    chain, head = build(7501)
    pid = chain.procs[0].pid
    print(f"stage pid={pid}")

    rtts = [head.ping() for _ in range(5)]
    print(f"心跳往返: {', '.join(f'{r:.2f}' for r in rtts)} ms "
          f"(中位 {sorted(rtts)[2]:.2f}ms)")

    pos, tok = 0, 7
    for i in range(20):
        head.step([tok], pos + i)
    print(f"每輪延遲基線(EWMA) = {head.lat_ms:.2f}ms -> 門檻 {head.deadline_ms():.0f}ms"
          f"（硬上限 15000ms）")

    print("\n[卡住] SIGSTOP 之後再發一個 step，看多久被判定")
    import os, signal
    os.kill(pid, signal.SIGSTOP)
    t0 = time.perf_counter()
    try:
        head.step([tok], pos + 100)
        print("  沒有被判定為掉線（不該發生）")
    except StageLost as e:
        dt = time.perf_counter() - t0
        print(f"  {dt*1e3:.0f}ms 後判定 kind={e.kind}")
        print(f"  訊息: {e}")
    finally:
        os.kill(pid, signal.SIGCONT)
    chain.stop(abort=True)

    print("\n[死掉] SIGKILL 之後再發一個 step")
    chain2, head2 = build(7521)
    pid2 = chain2.procs[0].pid
    for i in range(5):
        head2.step([tok], i)
    import os, signal
    os.kill(pid2, signal.SIGKILL)
    time.sleep(0.2)
    t0 = time.perf_counter()
    try:
        head2.step([tok], 50)
        print("  沒有被判定為掉線（不該發生）")
    except StageLost as e:
        dt = time.perf_counter() - t0
        print(f"  {dt*1e3:.0f}ms 後判定 kind={e.kind}")
        print(f"  訊息: {e}")
    chain2.stop(abort=True)


if __name__ == "__main__":
    main()
