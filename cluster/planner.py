"""Exact tiling planner under BOTH constraints: time and memory.

Three things this gets right that a naive "split it evenly" does not:

1. OBJECTIVE depends on the regime.
   objective="max"  stages run concurrently (speculative / pipelined)
   objective="sum"  stages run as a relay; sum is what you actually wait for
   With hop≈layer time, a relay gets SLOWER with every added node.

2. PER-STAGE FIXED COST. A stage is not `n * tau`; it is `fixed + n * tau`.
   Measured: 1 层 0.80ms, 4 层 2.14ms, 7 层 3.06ms -> fixed≈0.42ms, tau≈0.38ms.
   Without this term the planner promises speedups that the fixed cost eats.

3. MEMORY IS ASYMMETRIC. Weights and KV live on the node that owns the layers,
   and node 0 also carries embed + lm_head. A plan that is fast and does not
   fit is not a plan.
"""
from typing import List, Optional


def _prefix(vals):
    p = [0.0]
    for v in vals:
        p.append(p[-1] + v)
    return p


def plan(layer_costs: List[float], n_nodes: int, hop_ms: float = 0.0,
         node_speed: Optional[List[float]] = None, objective: str = "max",
         stage_fixed_ms: float = 0.0, mem: Optional[dict] = None):
    """Returns (bounds, cost_ms, info). bounds excludes idle nodes."""
    n = len(layer_costs)
    if objective not in ("max", "sum"):
        raise ValueError(objective)
    speed = list((node_speed or [1.0] * n_nodes)[:n_nodes])
    if len(speed) < n_nodes:
        speed += [speed[-1]] * (n_nodes - len(speed))

    pt = _prefix(layer_costs)
    pw = _prefix([mem["layer_w"]] * n) if mem else None
    pkv = _prefix([mem["layer_kv_token"]] * n) if mem else None

    def stage_ms(k, j, i):
        return stage_fixed_ms + (pt[i] - pt[j]) / speed[k] + hop_ms

    def stage_bytes(k, j, i):
        if k >= len(mem["capacity"]):
            return float("inf")
        b = pw[i] - pw[j] + (pkv[i] - pkv[j]) * mem["ctx"]
        if k == 0:
            b += mem.get("head", 0.0)
        return b

    def fits(k, j, i):
        if mem is None:
            return True
        if k >= len(mem["capacity"]):
            return False
        return stage_bytes(k, j, i) <= mem["capacity"][k]

    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(n_nodes + 1)]
    cut = [[0] * (n + 1) for _ in range(n_nodes + 1)]
    dp[0][0] = 0.0
    for k in range(1, n_nodes + 1):
        for i in range(1, n + 1):
            best, arg = INF, None
            for j in range(0, i + 1):
                if dp[k - 1][j] == INF:
                    continue
                if i == j:
                    cand = dp[k - 1][j]           # node k idle: no layers, no hop, no cost
                else:
                    if not fits(k - 1, j, i):
                        continue
                    seg = stage_ms(k - 1, j, i)
                    cand = max(dp[k - 1][j], seg) if objective == "max" \
                        else dp[k - 1][j] + seg
                if cand < best:
                    best, arg = cand, j
            dp[k][i] = best
            cut[k][i] = arg
    if dp[n_nodes][n] == INF:
        return None, INF, dict(feasible=False, reason="no feasible tiling",
                               min_nodes=None)

    bounds, i = [], n
    for k in range(n_nodes, 0, -1):
        j = cut[k][i]
        if i > j:
            bounds.append((j, i))
        i = j
    bounds.reverse()

    peak = 0.0
    if mem is not None:
        for k, (lo, hi) in enumerate(bounds):
            peak = max(peak, stage_bytes(k, lo, hi))
    return bounds, dp[n_nodes][n], dict(feasible=True, peak_bytes=peak)


def min_nodes(layer_costs, max_nodes=None, hop_ms=0.0, speed=None,
              stage_fixed_ms=0.0, mem=None, objective="max"):
    """Smallest node count that yields a feasible plan, or None.

    You cannot cap the search at the count you happen to be asking about: if 4
    nodes do not fit, the answer the operator needs is "you need 8", not "it is
    impossible".
    """
    ub = max_nodes or len(layer_costs)
    for k in range(1, ub + 1):
        b, c, info = plan(layer_costs, k, hop_ms, speed, objective,
                          stage_fixed_ms, mem)
        if b is not None:
            return k, b, c
    return None, None, float("inf")


def max_ctx(layer_costs, n_nodes, mem, objective="max", hop_ms=0.0,
            stage_fixed_ms=0.0, lo=1024, hi=131072):
    """Largest context that still fits on this many nodes (grows with nodes)."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m2 = dict(mem)
        m2["ctx"] = mid
        if len(m2.get("capacity", [])) != n_nodes:
            m2["capacity"] = [m2["capacity"][0]] * n_nodes
        b, _, _ = plan(layer_costs, n_nodes, hop_ms, None, objective,
                       stage_fixed_ms, m2)
        if b is not None:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def validate(layer_costs, bounds, mem=None):
    """Self-check the plan before any process is started.

    Catches the failure mode that has no error message: a tiling that is
    contiguous-looking but leaves a gap or an overlap, so every stage happily
    computes the wrong layers and the output is quietly wrong.
    """
    issues = []
    n = len(layer_costs)
    if not bounds:
        return ["no bounds at all"]
    if bounds[0][0] != 0:
        issues.append(f"first stage starts at layer {bounds[0][0]}, not 0")
    if bounds[-1][1] != n:
        issues.append(f"last stage ends at layer {bounds[-1][1]}, not {n}")
    for k in range(1, len(bounds)):
        if bounds[k][0] != bounds[k - 1][1]:
            issues.append(f"gap or overlap between n{k - 1}{list(bounds[k - 1])} "
                          f"and n{k}{list(bounds[k])}")
    for k, (lo, hi) in enumerate(bounds):
        if hi <= lo:
            issues.append(f"n{k} owns no layers ({lo}:{hi})")
    if mem:
        for k, (lo, hi) in enumerate(bounds):
            need = (hi - lo) * (mem["layer_w"] + mem["layer_kv_token"] * mem["ctx"])
            if k == 0:
                need += mem.get("head", 0.0)
            cap = mem["capacity"][min(k, len(mem["capacity"]) - 1)]
            if need > cap:
                issues.append(f"n{k} needs {need / 1e9:.1f}G of {cap / 1e9:.1f}G")
    return issues


def describe(cfg_layers, bounds):
    return " -> ".join(f"n{i}[{lo}:{hi}]" for i, (lo, hi) in enumerate(bounds))
