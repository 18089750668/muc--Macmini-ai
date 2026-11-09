"""Where does the 3.1ms per round actually go?

Splits the cost into independent pieces so we optimise the real one:
  A. raw TCP round-trip (echo child), sizes 48B / 8KB / 64KB
  B. CPU framing: JSON header vs 48-byte struct header
  C. crc32 over payload
  D. numpy / MLX conversions

Usage: transport_micro.py            (parent)
       transport_micro.py --echo N   (child, internal)
"""
import json
import socket
import statistics
import struct
import subprocess
import sys
import time
import zlib

import numpy as np

HOST = "127.0.0.1"
NODELAY = socket.TCP_NODELAY


def _nodelay(s):
    s.setsockopt(socket.IPPROTO_TCP, NODELAY, 1)


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if not r:
            raise ConnectionError("closed")
        got += r
    return buf


def echo_child(port, size):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    conn, _ = srv.accept()
    _nodelay(conn)
    while True:
        buf = _recv_exact(conn, size)
        if not buf:
            break
        conn.sendall(buf)


def raw_rtt(port, size, iters=200):
    s = socket.create_connection((HOST, port))
    _nodelay(s)
    payload = b"\x00" * size
    times = []
    for i in range(iters):
        t0 = time.perf_counter()
        s.sendall(payload)
        _recv_exact(s, size)
        times.append((time.perf_counter() - t0) * 1e3)
    s.close()
    return statistics.median(times), min(times)


def cpu_costs(payload_bytes):
    meta = {"start_pos": 128, "chunk_id": 77, "n_in": 5, "req": 1,
            "shape": [5, 256], "crc": 12345, "nbytes": payload_bytes}
    arr = np.zeros((5, 256), dtype=np.float32)
    buf = arr.tobytes()

    def bench(fn, iters=2000):
        for _ in range(20):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1e6      # us

    out = {}
    out["json_encode"] = bench(lambda: json.dumps(meta).encode())
    out["json_decode"] = bench(lambda: json.loads(json.dumps(meta).encode()))
    hdr = struct.pack("<4sHHIIIhBBIIIII", b"CLST", 1, 0, 1, 77, 128, 5, 2, 0,
                      *[5, 256, 1], len(buf), 0)
    out["struct_pack"] = bench(lambda: struct.pack(
        "<4sHHIIIhBBIIIII", b"CLST", 1, 0, 1, 77, 128, 5, 2, 0, 5, 256, 1, len(buf), 0))
    out["struct_unpack"] = bench(lambda: struct.unpack("<4sHHIIIhBBIIIII", hdr))
    out["crc32_8k"] = bench(lambda: zlib.crc32(buf))
    out["np_tobytes"] = bench(lambda: arr.tobytes())
    out["np_frombuffer_copy"] = bench(lambda: np.frombuffer(buf, dtype=np.float32).reshape(5, 256).copy())
    out["np_frombuffer_view"] = bench(lambda: np.frombuffer(buf, dtype=np.float32).reshape(5, 256))
    try:
        import mlx.core as mx
        out["mx_from_np_8k"] = bench(lambda: mx.array(np.frombuffer(buf, dtype=np.float32).reshape(5, 256)))
        out["np_from_mx_8k"] = bench(lambda: np.array(mx.array(np.frombuffer(buf, dtype=np.float32).reshape(5, 256))))
    except ImportError:
        pass
    return out


def parent():
    sizes = (48, 8192, 65536)
    procs, ports = [], []
    for i, size in enumerate(sizes):
        port = 5300 + i
        p = subprocess.Popen([sys.executable, __file__, "--echo", str(port), str(size)])
        procs.append(p)
        ports.append(port)
    time.sleep(0.6)

    print("A. raw TCP round-trip (bytes out + back)")
    print(f"{'size':>8} {'median_ms':>10} {'min_ms':>8}")
    for size, port in zip(sizes, ports):
        med, mn = raw_rtt(port, size)
        print(f"{size:>8} {med:>10.3f} {mn:>8.3f}")
    for p in procs:
        p.terminate()

    print("\nB/C/D. CPU-side cost per round (us)")
    costs = cpu_costs(8192)
    for k, v in costs.items():
        print(f"  {k:>24} {v:>9.2f} us")

    print("\n預測：每圈 = 2×encode + RTT + 2×decode。把上面數字代入，")
    print("就能看出 3.1ms 是 TCP 往返、framing、還是轉換造成的。")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--echo":
        echo_child(int(sys.argv[2]), int(sys.argv[3]))
    else:
        parent()
