"""實體層：這麼多台到底怎麼接起來。

TB4 是點對點直連才有低延遲，不是「插上交換機就成網」。而基礎版 M4 mini 只有
3 個 TB4 口，每條線吃兩端各一個 → 每台最多接 3 條線。所以拓撲不是挑出來的，
是被口數逼出來的：

    n 台環狀 = n 條線，每台度數 2（留 1 個口給外設/管理）
    全互聯 = n(n-1)/2 條線，每台度數 n-1 → 8 台以上就物理不可能

這支配線同時決定層怎麼切（ring 上第 i 台拿第 i 段），以及每台要跑什麼指令。
--relay 加上去時，順便把「出門在外」的 publish/consume 也印出來。

    python -m cluster.topology --nodes 8
    python -m cluster.topology --nodes 16 --relay relay.example:8800
"""
import argparse

TB4_PORTS = 3


def ring_links(n):
    """環狀：i 連 i+1，最後一台連回第一台。"""
    return [(i, (i + 1) % n) for i in range(n)]


def full_mesh_links(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def degree_of(links, n):
    d = [0] * n
    for a, b in links:
        d[a] += 1
        d[b] += 1
    return d


def layer_bounds(n_layers, n):
    """盡量平均；餘數給前面的機器。head 額外背 embed + lm_head，所以它不多拿層。"""
    base, rem = divmod(n_layers, n)
    out, lo = [], 0
    for i in range(n):
        hi = lo + base + (1 if i < rem else 0)
        out.append((lo, hi))
        lo = hi
    return out


def commands(n, n_layers, base_port, head_host, hosts=None):
    """head 只連第一台；之後每台連下一台；最後一台直接把結果回報 head。

    端口約定：stage 用 base+1 .. base+n，head 用 base+n+1 —— 兩段必須完全錯開。
    舊版把 head 放在 base+10，16 台時正好撞上第 10 台的 stage 端口，這裡修掉。
    """
    hosts = hosts or [f"m{i}" for i in range(n)]
    b = layer_bounds(n_layers, n)
    stage_port = {i: base_port + i for i in range(1, n)}
    head_port = base_port + n + 1
    cmds = [f"  [{hosts[0]} / head] python -m cluster.main run --nodes {n} "
            f"--peers {hosts[1]}:{stage_port[1]} "
            f"--advertise {head_host}:{head_port} "
            f"--head-port {head_port} --prompt-len 512 --max-tokens 200"]
    for i in range(1, n):
        lo, hi = b[i]
        tail = (f"--reply {head_host}:{head_port}" if i == n - 1
                else f"--next {hosts[i + 1]}:{stage_port[i + 1]}")
        cmds.append(f"  [{hosts[i]}] python -m cluster.main stage --layers {lo}:{hi} "
                    f"--port {stage_port[i]} {tail} --persist")
    return cmds, stage_port, head_port


def remote_commands(n, hosts, relay, stage_port):
    """出門在外：家裡每台 publish 自己的 stage 端口，筆記本 consume 成
    127.0.0.1:<本地埠>，於是 head 完全不用改（遠端看起來就是本機埠）。"""
    pub = [f"  [{hosts[i]}] publish --relay {relay} --name {hosts[i]} "
           f"--to 127.0.0.1:{stage_port[i]} --wait 300" for i in range(1, n)]
    con = [f"  consume --relay {relay} --listen {9100 + i} --name {hosts[i]}"
           for i in range(1, n)]
    return pub, con


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--layers", type=int, default=80)
    ap.add_argument("--base-port", type=int, default=7600)
    ap.add_argument("--head", default="192.168.1.10")
    ap.add_argument("--hosts", default=None, help="逗號分隔的主機名")
    ap.add_argument("--relay", default=None, help="中轉機 host:port，給了就印遠端指令")
    a = ap.parse_args(argv)

    n = a.nodes
    hosts = a.hosts.split(",") if a.hosts else [f"m{i}" for i in range(n)]
    if len(hosts) != n:
        raise SystemExit(f"--hosts 給了 {len(hosts)} 個名字，但 --nodes={n}")

    ring = ring_links(n)
    mesh = full_mesh_links(n)
    d_ring = degree_of(ring, n)
    d_mesh = degree_of(mesh, n)

    print(f"{n} 台：环状 = {len(ring)} 条线，每台度数 {max(d_ring)}"
          f"（可用 {TB4_PORTS} 个 TB4 口，剩 {TB4_PORTS - max(d_ring)} 个给外设/管理）")
    print(f"       全互联需要 {len(mesh)} 条线、度数 {max(d_mesh)} -> "
          f"{'可行' if max(d_mesh) <= TB4_PORTS else '不可行'}")

    print("\n布线（每条线两端各占一个 TB4 口；同一对机器之间不要串交换机，"
          "TB4 是点对点直连才有低延迟）：")
    for i, j in ring:
        print(f"  {hosts[i]} --- {hosts[j]}")

    print("\n每台的 IP 建议：TB4 网桥各自独立 L2，用静态 /30 或链路本地 "
          "169.254.x，不要依赖 DHCP；网段随线走，机器换口要重配。")

    b = layer_bounds(a.layers, n)
    chain = " -> ".join(f"{hosts[i]}[{lo}:{hi}]" for i, (lo, hi) in enumerate(b))
    print(f"\n层切分（{a.layers} 层 / {n} 台）：{chain}")

    cmds, stage_port, head_port = commands(n, a.layers, a.base_port, a.head, hosts)
    print("\n每台机器的启动命令（node 0 不用跑 stage，它是 head）：")
    for c in cmds:
        print(c)

    if a.relay:
        pub, con = remote_commands(n, hosts, a.relay, stage_port)
        print(f"\n出门在外（中转机 {a.relay}）：家里先 publish，笔记本再 consume，"
              f"然后 head 直接把 127.0.0.1:<本地埠> 当 --peers：")
        for c in pub:
            print(c)
        print("")
        for c in con:
            print(c)
        print(f"\n  笔记本 head：python -m cluster.main run --nodes {n} "
              f"--peers 127.0.0.1:{9100 + 1} --advertise 127.0.0.1:9200 "
              f"--head-port 9200 ...")
        print("  （反向那条：笔记本 publish --name headin --to 127.0.0.1:9200，"
              "家里最后一段的 --reply 指向本地 consume headin）")
        print("  开机自启：python -m cluster.remote plist --relay "
              f"{a.relay} --name {hosts[1]} --to-port {stage_port[1]}")


if __name__ == "__main__":
    main()
