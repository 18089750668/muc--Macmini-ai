"""Real network round trip with the v2 protocol: is 0.071ms a loopback number only?

STATUS: unverified. The only second machine available here is a cloud VM whose
address is not reachable from the Mac (the Mac's connect() is answered by a local
proxy that accepts and then black-holes the payload; the VM never sees an accept).
Run this once a real TB4/LAN peer exists.

Connects to an echoing stage over an actual network and times send_frame ->
recv_frame. The echo server need not understand the model; it just returns the
frame bytes, which is exactly what a stage would do to a HEARTBEAT.

    python bench/cross_machine_rtt.py --host 9.135.17.187 --port 7801
"""
import argparse
import statistics
import time

import numpy as np

from cluster import protocol as P


def rtt(sock, payload_bytes, reps):
    arr = np.zeros(payload_bytes // 4, dtype=np.float32)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        P.send_frame(sock, P.HEARTBEAT, {"req": 1}, arr)
        ftype, meta, got = P.recv_frame(sock)
        ts.append((time.perf_counter() - t0) * 1e3)
        assert got is not None and got.shape == arr.shape, (got is None, got.shape)
    return statistics.median(ts), min(ts)


def main(argv=None):
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=7801)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args(argv)
    sock = P.connect(a.host, a.port, retries=10)
    P.tune(sock)
    for nbytes in (0, 8192, 65536):
        med, mn = rtt(sock, nbytes, a.reps)
        print(f"  {nbytes:6d} B: median {med:7.2f} ms   min {mn:7.2f} ms")
    sock.close()


if __name__ == "__main__":
    main()
