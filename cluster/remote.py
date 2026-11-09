"""Remote access for the cluster: let a laptop reach home machines over the
internet, with no port forwarding, and no re-auth after reboot.

Why this shape and not "just open a port":

  * Home machines sit behind NAT (often CGNAT in CN). No inbound port exists to
    open, so the only way in is for the home side to dial OUT and keep the
    connection alive.
  * So there is one small relay (a cloud VM, or any box with a public IP). Home
    machines *publish* themselves to it; the laptop *consumes* them. Both
    directions of the model protocol (head -> stage, tail -> head) are handled by
    the same two primitives, so stage.py / coordinator.py need no change at all:
    remote endpoints simply appear as 127.0.0.1 ports.

  * Credentials live in ~/.cluster/token (0600). The publish/bridge loops re-dial
    forever, so after a reboot (or a laptop that comes home) nothing needs to be
    typed: launchd starts the bridge at login, it reconnects, done.

Primitives:

    gateway  --port 8800                         # on the relay box (public IP)
    publish  --relay H:P --name m1 --to 127.0.0.1:7001   # on a home machine
    consume  --relay H:P --listen 127.0.0.1:9001 --name m1 # on the laptop

Wire it up:

    # home (each machine, in launchd so it starts at boot):
    remote publish --relay relay.example:8800 --name m1 --to 127.0.0.1:7001
    # laptop:
    remote consume --relay relay.example:8800 --listen 127.0.0.1:9001 --name m1
    cluster.main run --nodes 2 --peers 127.0.0.1:9001 ...

The reverse path (tail -> head) uses the same pair in the other direction: publish
"headin" -> 127.0.0.1:<head-port> on the laptop, and point the tail's --reply at a
local consume of "headin".

SECURITY, plainly: this relay moves bytes with no encryption. A stolen token is
full access to your models. Run the gateway inside WireGuard/Tailscale, or put
TLS in front of it, before exposing it to the open internet. The token is
authentication, not a substitute for a private network.
"""
import argparse
import base64
import hmac
import json
import os
import socket
import struct
import sys
import threading
import time

TOKEN_PATH = os.path.expanduser("~/.cluster/token")
BUF = 65536


# ---------------------------------------------------------------- credentials
def load_token(explicit=None):
    if explicit:
        return explicit
    env = os.environ.get("CLUSTER_TOKEN")
    if env:
        return env
    try:
        with open(TOKEN_PATH) as f:
            return f.read().strip()
    except OSError:
        raise SystemExit(
            f"no token: put it in {TOKEN_PATH} (chmod 600), or pass --token, "
            f"or set CLUSTER_TOKEN")


def new_token(path=TOKEN_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok = base64.urlsafe_b64encode(os.urandom(24)).decode().rstrip("=")
    with open(path, "w") as f:
        f.write(tok + "\n")
    os.chmod(path, 0o600)
    return tok


def tune(sock, keepalive=False):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if not keepalive:
        return
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, val in (("TCP_KEEPALIVE", 30), ("TCP_KEEPINTVL", 10),
                     ("TCP_KEEPCNT", 6)):
        o = getattr(socket, opt, None)
        if o is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, o, val)
            except OSError:
                pass


# --------------------------------------------------------------- hello wiring
def _hello(role, name, token):
    return (b"GW " + json.dumps({"role": role, "name": name, "token": token}
                                ).encode() + b"\n")


def _readline(sock, limit=4096):
    buf = b""
    while b"\n" not in buf and len(buf) < limit:
        chunk = sock.recv(1)
        if not chunk:
            return None
        buf += chunk
    return buf


def _send_two(a, b):
    """Byte-level pump between two sockets; returns when either side is done."""
    def one_way(src, dst):
        try:
            while True:
                data = src.recv(BUF)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    t = threading.Thread(target=one_way, args=(b, a), daemon=True)
    t.start()
    one_way(a, b)
    t.join(timeout=5)


# --------------------------------------------------------------------- gateway
class _Pair:
    """一次配對的兩端：sock = 註冊進來的 socket，peer = 配對到的另一端。"""

    __slots__ = ("sock", "ev", "peer")

    def __init__(self, sock):
        self.sock = sock
        self.peer = None
        self.ev = threading.Event()


class _Pending:
    """name -> _Pair，單一登記位。

    規則：誰先到誰掛號（同名只留一個，後到的蓋掉先到的老屍體），另一方到達時
    把兩者湊成一對並 set Event。掛號的一方等到 Event 才回 OK，所以「登記了但
    根本沒連上自己的後端」不會被配對出去 —— 那會讓對面 write 到死 socket，
    表現成毫無線索的 broken pipe。
    """

    def __init__(self):
        self._cv = threading.Condition()
        self.pool = {}

    def put(self, name, sock):
        pair = _Pair(sock)
        with self._cv:
            old = self.pool.get(name)
            if old is not None and not old.ev.is_set():
                # 老屍體：它的 promise 已經過期，直接丟掉（對面會自己重試）
                try:
                    old.sock.close()
                except OSError:
                    pass
            self.pool[name] = pair
            self._cv.notify_all()
        return pair

    def pair(self, name, sock, timeout):
        """把 sock 跟 name 的掛號者配對；回傳對面的 _Pair，或 None（超時）。"""
        end = time.time() + timeout
        with self._cv:
            while name not in self.pool:
                left = end - time.time()
                if left <= 0:
                    return None
                self._cv.wait(left)
            other = self.pool.pop(name)
        other.peer = sock
        other.ev.set()
        return other


def cmd_gateway(a):
    token = load_token(a.token)

    def handle(c):
        pending = a._pending
        try:
            tune(c, keepalive=True)
            line = _readline(c)
            if not line:
                return
            try:
                msg = json.loads(line[3:].decode())
            except Exception:
                c.sendall(b"ERR malformed\n")
                return
            if not _eq(msg.get("token"), token):
                c.sendall(b"ERR bad token\n")      # 不給重試空間，也不回細節
                return
            name, role = str(msg.get("name")), msg.get("role")
            if role == "publish":
                mine = pending.put(name, c)         # 掛號，等 consumer 來配
                if not mine.ev.wait(a.wait):
                    c.sendall(b"ERR no consumer\n")
                    return
                c.sendall(b"OK\n")                 # 配對成功才放行
                _send_two(c, mine.peer)
            elif role == "consume":
                other = pending.pair(name, c, a.wait)
                if other is None:
                    c.sendall(b"ERR no publisher\n")
                    return
                c.sendall(b"OK\n")
                return                              # socket 交給 publisher 的 pump
            else:
                c.sendall(b"ERR bad role\n")
        except OSError:
            pass

    a._pending = _Pending()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(64)
    print(f"[gateway] listening {a.host}:{a.port}  (token from {TOKEN_PATH})",
          flush=True)
    while True:
        c, addr = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def _eq(a, b):
    """定時比較：避免用回應時間猜 token。"""
    if a is None:
        return False
    return hmac.compare_digest(a.encode(), b.encode())


# --------------------------------------------------------------------- publish
def cmd_publish(a):
    token = load_token(a.token)
    while True:
        try:
            # 先確定自己的後端還活著，再去 relay 掛號：順序反了就會出現
            # 「掛了號但後端已經死」，配對出去的那條連線一寫就 broken pipe。
            local = socket.create_connection((a.to_host, a.to_port), timeout=a.dial)
            tune(local)
            g = socket.create_connection((a.relay_host, a.relay_port), timeout=a.dial)
            tune(g, keepalive=True)
            g.settimeout(a.wait + 10)
            g.sendall(_hello("publish", a.name, token))
            reply = _readline(g)
            if reply != b"OK\n":
                raise OSError(f"relay refused: {reply!r}")
            g.settimeout(None)
            print(f"[publish] {a.name}: relay -> 127.0.0.1:{a.to_port}", flush=True)
            _send_two(g, local)
            local.close()
            g.close()
        except OSError as e:
            print(f"[publish] {a.name}: {e}; retry in {a.retry}s", flush=True)
            time.sleep(a.retry)


def cmd_consume(a):
    token = load_token(a.token)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.listen))
    srv.listen(8)
    print(f"[consume] 127.0.0.1:{a.listen} -> {a.name} @ relay", flush=True)

    def handle(c):
        try:
            tune(c)
            g = socket.create_connection((a.relay_host, a.relay_port), timeout=a.dial)
            tune(g, keepalive=True)
            g.settimeout(a.wait + 10)
            g.sendall(_hello("consume", a.name, token))
            reply = _readline(g)
            if reply != b"OK\n":
                raise OSError(f"relay refused: {reply!r}")
            g.settimeout(None)
            _send_two(c, g)
            g.close()
        except OSError as e:
            print(f"[consume] {a.name}: {e}", flush=True)
        finally:
            c.close()

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


# ------------------------------------------------------- launchd (starts at boot)
PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.localcluster.{name}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string>
    <string>-m</string><string>cluster.remote</string><string>publish</string>
    <string>--relay</string><string>{relay}</string>
    <string>--name</string><string>{name}</string>
    <string>--to</string><string>127.0.0.1:{to_port}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>WorkingDirectory</key><string>{root}</string>
  <key>StandardOutPath</key><string>/tmp/cluster-{name}.log</string>
  <key>StandardErrorPath</string><string>/tmp/cluster-{name}.log</string>
</dict>
</plist>
"""


def cmd_plist(a):
    to_port = a.to_port or 7001
    print(PLIST.format(python=sys.executable, name=a.name, relay=a.relay,
                       to_port=to_port, root=a.root))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cluster.remote")
    ap.add_argument("--token", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gateway")
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--port", type=int, default=8800)
    g.add_argument("--wait", type=float, default=30.0,
                   help="秒：publisher 等 consumer 配對的上限")
    g.set_defaults(fn=cmd_gateway)

    p = sub.add_parser("publish")
    p.add_argument("--relay", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--to", dest="to_addr", default="127.0.0.1:7001")
    p.add_argument("--retry", type=float, default=5.0)
    p.add_argument("--dial", type=float, default=10.0)
    p.add_argument("--wait", type=float, default=30.0,
                   help="秒：掛號後等 consumer 過來的上限")
    p.add_argument("--root", default=".")
    p.set_defaults(fn=cmd_publish)

    c = sub.add_parser("consume")
    c.add_argument("--relay", required=True)
    c.add_argument("--listen", type=int, required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--dial", type=float, default=10.0)
    c.add_argument("--wait", type=float, default=30.0)
    c.set_defaults(fn=cmd_consume)

    pl = sub.add_parser("plist")
    pl.add_argument("--relay", required=True)
    pl.add_argument("--name", required=True)
    pl.add_argument("--to-port", type=int, default=7001)
    pl.add_argument("--root", default=os.getcwd())
    pl.set_defaults(fn=cmd_plist)

    t = sub.add_parser("newtoken")
    t.set_defaults(fn=lambda a: print(new_token()))

    a = ap.parse_args(argv)
    for attr in ("relay", "to_addr", "to_port"):
        v = getattr(a, attr, None)
        if isinstance(v, str) and ":" in v:
            h, pt = v.rsplit(":", 1)
            if attr == "relay":
                a.relay_host, a.relay_port = h, int(pt)
            else:
                a.to_host, a.to_port = h, int(pt)
    a.fn(a)


if __name__ == "__main__":
    main()
