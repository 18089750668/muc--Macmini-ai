"""Transport-only round-trip using the REAL protocol module (no model).

Answers one question precisely: after the v2 upgrade, what does one pipe stage
hop cost on the wire, with nothing else in the loop?
"""
import statistics
import sys
import socket
import time

import numpy as np

from cluster import protocol as P

HOST = "127.0.0.1"


def child(port, spin):
    P.set_spin(spin)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    conn, _ = srv.accept()
    P.tune(conn)
    while True:
        P.arm(conn)
        try:
            ft, meta, arr = P.recv_frame(conn)
        except ConnectionError:
            break
        if ft == P.ABORT:
            break
        P.send_frame(conn, P.RESULT, meta, np.zeros(5, dtype=np.int32))


def parent(port, spin, iters=500):
    P.set_spin(spin)
    s = P.connect(HOST, port, retries=60)
    act = np.zeros((1, 256), dtype=np.float32)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        P.send_frame(s, P.VERIFY, {"chunk_id": 1, "start_pos": 0, "n_in": 1}, act)
        P.arm(s)
        P.recv_frame(s)
        times.append((time.perf_counter() - t0) * 1e3)
    P.send_frame(s, P.ABORT, {})
    return statistics.median(times), min(times)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(int(sys.argv[2]), int(sys.argv[3]))
    else:
        import subprocess
        print(f"{'spin_us':>8} {'median_ms':>10} {'min_ms':>8}")
        for i, spin in enumerate((0, 2000, 4000)):
            port = 5400 + i
            p = subprocess.Popen([sys.executable, __file__, "--child", str(port), str(spin)])
            time.sleep(0.5)
            med, mn = parent(port, spin)
            print(f"{spin:>8} {med:>10.3f} {mn:>8.3f}")
            p.terminate()
