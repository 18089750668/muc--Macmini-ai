"""實體層：這麼多台到底怎麼接起來。

TB4 是點對點直連才有低延遲，不是「插上交換機就成網」。而基礎版 M4 mini 只有
3 個 TB4 口，每條線吃兩端各一個，所以「每台最多接 3 條線」。實際佈線是**鏈**，
不是環、更不是全互聯：

    鏈   = n-1 條線；端點度數 1、中間每台度數 2（還剩 1 個口給電源/管理/外設）
    環   = n   條線；每台度數 2，但要多拉一條「尾巴直連頭」的線
    全互聯 = n(n-1)/2 條線、每台度數 n-1 → 4 台以上就物理不可能

為什麼這件事會影響程式：鏈上沒有「一步回得到頭」的路。最後一台算完的結果
只能交給隔壁，一級一級傳上去，所以每台要多開一個**回程口**（--ret-port），
它的 --reply 指向上游那一台的回程口（最前面那台指 head）。

這支配線同時決定層怎麼切（第 i 台拿第 i 段），以及每台要跑什麼指令。

    python -m cluster.topology --nodes 4
    python -m cluster.topology --nodes 16 --relay relay.example:8800
"""
import argparse

TB4_PORTS = 3


def chain_links(n):
    """鏈：0-1-2-...-(n-1)。n 台就是 n-1 條線，這是實體佈線。"""
    return [(i, i + 1) for i in range(n - 1)]


def ring_links(n):
    """環：多接一條「尾巴直連頭」。只有在你真的願意多拉那一條線時才用。"""
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


def ports_for(n, base_port):
    """stage 用 base+1..base+n，head 用 base+n+1，回程口用 base+n+2 起。

    三段必須完全錯開：舊版把 head 放在 base+10，16 台時正好撞上第 10 台的
    stage 埠，連到殘留進程、形狀合法但輸出悄悄錯掉。
    """
    stage = {i: base_port + i for i in range(1, n)}
    head = base_port + n + 1
    ret = {i: base_port + n + 1 + i for i in range(1, n - 1)}   # 中間那幾台才要
    return stage, head, ret


def commands(n, n_layers, base_port, head_host, hosts=None):
    """鏈式佈線的完整啟動指令（每台一行）。

    第 i 台的 `--reply` 指「上游那一台的回程口」（第 1 台指 head）；
    只有「下面還有人」的台才帶 `--ret-port`。head 只連第 1 台，
    結果從第 1 台的回程口送上來。
    """
    hosts = hosts or [f"m{i}" for i in range(n)]
    b = layer_bounds(n_layers, n)
    stage_port, head_port, ret_port = ports_for(n, base_port)
    cmds = [f"  [{hosts[0]} / head] python -m cluster.main run --nodes {n} "
            f"--peers {hosts[1]}:{stage_port[1]} "
            f"--advertise {head_host}:{head_port} "
            f"--head-port {head_port} --chain --prompt-len 512 --max-tokens 200"]
    for i in range(1, n):
        lo, hi = b[i]
        # 回程口在「上游那一台」機器上，所以要用那台的位址，不是 127.0.0.1
        back = (f"{head_host}:{head_port}" if i == 1
                else f"{hosts[i - 1]}:{ret_port[i - 1]}")
        cmd = (f"  [{hosts[i]}] python -m cluster.main stage --layers {lo}:{hi} "
               f"--port {stage_port[i]} --reply {back} --chain --persist")
        if i < n - 1:
            cmd += (f" --next {hosts[i + 1]}:{stage_port[i + 1]}"
                    f" --ret-port {ret_port[i]}")
        cmds.append(cmd)
    return cmds, stage_port, head_port


def remote_commands(n, hosts, relay, stage_port, ret_port=None, head_port=None):
    """出門在外：家裡每台 publish 自己的 stage 端口，筆記本 consume 成
    127.0.0.1:<本地埠>，於是 head 完全不用改（遠端看起來就是本機埠）。

    鏈式多一件事：第 1 台的回程口要「反向」被筆記本 consume，因為它的
    --reply 指向 head。做法是家裡把那台的回程口 publish 成 `headin`，
    筆記本 consume 成 head 的監聽埠。
    """
    pub = [f"  [{hosts[i]}] publish --relay {relay} --name {hosts[i]} "
           f"--to 127.0.0.1:{stage_port[i]} --wait 300" for i in range(1, n)]
    con = [f"  consume --relay {relay} --listen {9100 + i} --name {hosts[i]}"
           for i in range(1, n)]
    if ret_port and head_port:
        pub.append(f"  [{hosts[1]}] publish --relay {relay} --name headin "
                   f"--to 127.0.0.1:{ret_port[1]} --wait 300")
        con.append(f"  consume --relay {relay} --listen {head_port} --name headin")
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

    link = chain_links(n)
    ring = ring_links(n)
    mesh = full_mesh_links(n)
    d_link, d_ring, d_mesh = (degree_of(link, n), degree_of(ring, n),
                              degree_of(mesh, n))

    inner = d_link[1] if n > 2 else d_link[0]
    print(f"{n} 台：链 = {len(link)} 条线（= {n}-1），端点度数 {d_link[0]}、"
          f"中间度数 {inner}（可用 {TB4_PORTS} 个 TB4 口，"
          f"剩 {TB4_PORTS - inner} 个给电源/管理/外设）")
    print(f"       环 = {len(ring)} 条线、每台度数 {max(d_ring)}（多拉一条尾巴→头的线）")
    print(f"       全互联 = {len(mesh)} 条线、度数 {max(d_mesh)} -> "
          f"{'可行' if max(d_mesh) <= TB4_PORTS else '物理不可能'}")
    print("       本專案預設走**鏈**：少一條線、每台只接上下家，"
          "代價是最後一台的結果要多跳幾次才回到 head。")

    print("\n布线（每条线两端各占一个 TB4 口；同一对机器之间不要串交换机，"
          "TB4 是点对点直连才有低延迟）：")
    for i, j in link:
        print(f"  {hosts[i]} --- {hosts[j]}")

    print("\n每台的 IP 建议：TB4 网桥各自独立 L2，用静态 /30 或链路本地 "
          "169.254.x，不要依赖 DHCP；网段随线走，机器换口要重配。")

    b = layer_bounds(a.layers, n)
    lay = " -> ".join(f"{hosts[i]}[{lo}:{hi}]" for i, (lo, hi) in enumerate(b))
    print(f"\n层切分（{a.layers} 层 / {n} 台）：{lay}")

    cmds, stage_port, head_port = commands(n, a.layers, a.base_port, a.head, hosts)
    print("\n每台机器的启动命令（node 0 不用跑 stage，它是 head）：")
    for c in cmds:
        print(c)

    if a.relay:
        _, _, ret_port = ports_for(n, a.base_port)
        pub, con = remote_commands(n, hosts, a.relay, stage_port, ret_port,
                                   a.base_port + n + 1)
        print(f"\n出门在外（中转机 {a.relay}）：家里先 publish，笔记本再 consume，"
              f"然后 head 直接把 127.0.0.1:<本地埠> 当 --peers：")
        for c in pub:
            print(c)
        print("")
        for c in con:
            print(c)
        print(f"\n  笔记本 head：python -m cluster.main run --nodes {n} "
              f"--peers 127.0.0.1:9101 "
              f"--advertise 127.0.0.1:{a.base_port + n + 1} "
              f"--head-port {a.base_port + n + 1} --chain ...")
        print("  （headin 那一條就是鏈的回程：家里第 1 台的回程口 -> 笔记本的 head）")
        print("  开机自启：python -m cluster.remote plist --relay "
              f"{a.relay} --name {hosts[1]} --to-port {stage_port[1]}")


if __name__ == "__main__":
    main()
