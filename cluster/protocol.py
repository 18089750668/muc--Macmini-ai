"""Wire format v2: fixed 48-byte struct header + raw payload, ONE sendall per
frame, ONE slab read per frame.

v1 was 3 sendalls + 3 blocking recvs per direction (4-byte length, JSON header,
payload) with Nagle on. The cost was not bandwidth (8KB round-trip is 60us) and
not framing (struct pack is 0.3us) -- it was the wake-ups: every partial read
blocks the process again. Hence this design: one write, one read, parse in place.

Integrity: crc32 over the payload. This catches transport corruption, NOT a
dishonest peer -- that distinction is the whole point of the Verification layer.
"""
import select
import json
import socket
import struct
import time
import zlib

import numpy as np

MAGIC = b"CLST"
_HEADER = struct.Struct("<4sHHIIihBBIIIIIf")   # 48 bytes
assert _HEADER.size == 48, _HEADER.size

PREFILL = "PREFILL"
VERIFY = "VERIFY"
RESULT = "RESULT"
HEARTBEAT = "HEARTBEAT"
ABORT = "ABORT"
RESET = "RESET"

_TYPE_ID = {PREFILL: 1, VERIFY: 2, RESULT: 3, HEARTBEAT: 4, ABORT: 5, RESET: 6}
_ID_TYPE = {v: k for k, v in _TYPE_ID.items()}
_DTYPE_ID = {"<f4": 0, "<f2": 1, "<i4": 2}
_ID_DTYPE = {v: k for k, v in _DTYPE_ID.items()}
_SLAB = 1 << 17          # 128 KiB: any of our frames lands in one read


def tune(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


_SPIN_US = 0
_spin_us = 0


def set_spin(us):
    global _spin_us
    _spin_us = int(us)


def arm(sock, us=None):
    """Tell the reader to poll instead of sleeping for the next `us` microseconds.
    Sleeping costs 0.5ms per wake-up on macOS; polling costs CPU. For a latency
    benchmark the trade is worth it, and it is exactly the knob a real
    implementation turns when it moves from syscalls to kernel bypass."""
    r = _reader(sock)
    window = _spin_us if us is None else us
    r.spin_until = time.perf_counter() + window * 1e-6


class _Reader:
    __slots__ = ("sock", "buf", "view", "start", "end", "spin_until")

    def __init__(self, sock):
        self.sock = sock
        self.buf = bytearray(_SLAB)
        self.view = memoryview(self.buf)
        self.start = 0
        self.end = 0
        self.spin_until = 0.0

    def _wait(self):
        if time.perf_counter() < self.spin_until:
            while True:
                if select.select([self.sock], [], [], 2e-5)[0]:
                    return
                if time.perf_counter() >= self.spin_until:
                    break
        t = self.sock.gettimeout()
        if t is not None:
            if not select.select([self.sock], [], [], t)[0]:
                raise socket.timeout("timed out waiting for a frame")
            return
        select.select([self.sock], [], [], None)

    def _need(self, n):
        while self.end - self.start < n:
            if self.end == len(self.buf):
                if self.start:
                    self.view.release()
                    self.buf[: self.end - self.start] = self.buf[self.start : self.end]
                    self.end -= self.start
                    self.start = 0
                    self.view = memoryview(self.buf)
                if self.end == len(self.buf):
                    self.view.release()
                    self.buf.extend(bytearray(len(self.buf)))
                    self.view = memoryview(self.buf)
            self._wait()
            got = self.sock.recv_into(self.view[self.end :], len(self.buf) - self.end)
            if got == 0:
                raise ConnectionError("peer closed")
            self.end += got

    def frame(self):
        self._need(_HEADER.size)
        (magic, tid, extra_len, req, chunk, start_pos, n_in, ndim, did,
         d0, d1, d2, nbytes, crc, tau) = _HEADER.unpack_from(self.buf, self.start)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        self._need(_HEADER.size + nbytes + extra_len)

        off = self.start + _HEADER.size
        payload = memoryview(self.buf)[off : off + nbytes] if nbytes else None
        if nbytes and zlib.crc32(payload) != crc:
            raise ValueError("checksum mismatch")

        arr = None
        if nbytes:
            shape = tuple((d0, d1, d2)[:ndim])
            arr = np.frombuffer(payload, dtype=_ID_DTYPE[did]).reshape(shape)

        meta = dict(req=req, chunk_id=chunk, start_pos=start_pos,
                    n_in=n_in, tau_ms=tau)
        if extra_len:
            blob = bytes(memoryview(self.buf)[off + nbytes: off + nbytes + extra_len])
            meta.update(json.loads(blob))
        self.start = off + nbytes + extra_len
        if self.start == self.end:
            self.start = self.end = 0

        return _ID_TYPE[tid], meta, arr


_readers = {}


def _reader(sock):
    # 快取按 id(sock) 存。id 只有在舊物件被回收之後才會重用，而快取本身
    # 握著舊 socket 的強引用、舊物件又永遠不會回收 —— 兩者湊在一起就是
    # 「同一個 id 指到兩個不同的 socket」。一旦撞上，讀到的就是上一輪
    # 殘留的字節（形狀合法、內容錯、不報錯）。所以這裡除了 id 還要比物件本身。
    r = _readers.get(id(sock))
    if r is not None and r.sock is not sock:
        r = None
    if r is None:
        r = _readers[id(sock)] = _Reader(sock)
    return r


def forget(sock):
    """socket 關掉之後把它的讀取緩衝丟掉（含 slab 裡沒讀完的殘留字節）。"""
    _readers.pop(id(sock), None)


_SCALAR_KEYS = ("req", "chunk_id", "start_pos", "n_in", "tau_ms")


def send_frame(sock, ftype, meta=None, array=None):
    """48 字節定長 header + payload + **可選的 JSON 尾巴**。

    尾巴是為 batch 加的：header 只裝得下五個純量，而 batch 需要 `valid_from`
    這種每列都不一樣的東西。原本這些鍵會被靜默丟掉（實測：2 台 batch 的結果
    跟單機不一樣，但沒有任何錯誤訊息）。所以用 header 的 flags 欄位存尾巴長度，
    沒有額外鍵時尾巴是空的，熱路徑一個字節都不多。
    """
    meta = meta or {}
    extra = {k: v for k, v in meta.items() if k not in _SCALAR_KEYS}
    blob = json.dumps(extra, separators=(",", ":")).encode() if extra else b""
    payload = b""
    ndim, did, dims, nbytes, crc = 0, 0, (1, 1, 1), 0, 0
    if array is not None:
        a = np.ascontiguousarray(np.asarray(array))
        did = _DTYPE_ID[a.dtype.str]
        payload = a.tobytes()
        nbytes = len(payload)
        crc = zlib.crc32(payload)
        ndim = a.ndim
        dims = tuple(list(a.shape) + [1] * (3 - a.ndim))
    hdr = _HEADER.pack(MAGIC, _TYPE_ID[ftype], len(blob), meta.get("req", 0),
                       meta.get("chunk_id", 0), meta.get("start_pos", 0),
                       meta.get("n_in", 0), ndim, did, dims[0], dims[1], dims[2],
                       nbytes, crc, float(meta.get("tau_ms", 0.0)))
    sock.sendall(hdr + payload + blob)      # ONE write


def recv_frame(sock):
    return _reader(sock).frame()


def connect(host, port, retries=40, delay=0.05):
    import time
    last = None
    for _ in range(retries):
        try:
            return tune(socket.create_connection((host, port), timeout=5))
        except OSError as exc:
            last = exc
            time.sleep(delay)
    raise ConnectionError(f"could not connect to {host}:{port}: {last}")
