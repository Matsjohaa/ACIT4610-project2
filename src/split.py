# Decode a permutation into ≤ n_vehicles routes.
from typing import List, Sequence, Callable
import numpy as np
from .constants import SPLIT_METHOD

INF = 1e100


def equal_split(perm: List[int], n_vehicles: int) -> List[List[int]]:
    """
    Baseline split: cut the permutation into ~equal-sized chunks.
    Capacity-agnostic. Returns at most n_vehicles routes.
    """
    n = len(perm)
    if n == 0:
        return []
    per = max(1, (n + n_vehicles - 1) // n_vehicles)
    routes = [perm[i:i + per] for i in range(0, n, per)]
    return routes[:n_vehicles]


def single_route_cost(perm: Sequence[int], i: int, j: int, M: np.ndarray) -> float:
    """
    Cost of serving perm[i..j] as a single route:
    depot -> perm[i] -> ... -> perm[j] -> depot
    """
    if i > j:
        return 0.0
    s = M[0, perm[i]]
    for t in range(i, j):
        s += M[perm[t], perm[t + 1]]
    s += M[perm[j], 0]
    return s


def dp_optimal_split(perm: List[int], n_vehicles: int, M: np.ndarray) -> List[List[int]]:
    """
    Dynamic programming split (capacity-agnostic):
    partition the permutation into ≤ n_vehicles segments minimizing total route cost.
    """
    n = len(perm)
    if n == 0:
        return []

    # precompute segment costs for all i..j
    seg = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            seg[i][j] = single_route_cost(perm, i, j, M)

    dp = [[INF] * (n + 1) for _ in range(n_vehicles + 1)]
    prv = [[-1] * (n + 1) for _ in range(n_vehicles + 1)]
    dp[0][0] = 0.0

    for v in range(1, n_vehicles + 1):
        for j in range(1, n + 1):
            best, arg = INF, -1
            for i in range(v - 1, j):  # ensure at least one customer per used route
                cand = dp[v - 1][i] + seg[i][j - 1]
                if cand < best:
                    best, arg = cand, i
            dp[v][j], prv[v][j] = best, arg

    # choose best number of routes 1..n_vehicles
    best_v, best_cost = 1, dp[1][n]
    for v in range(2, n_vehicles + 1):
        if dp[v][n] < best_cost:
            best_v, best_cost = v, dp[v][n]

    # reconstruct routes
    routes: List[List[int]] = []
    j = n
    v = best_v
    while v > 0 and j > 0:
        i = prv[v][j]
        routes.append(perm[i:j])
        j = i
        v -= 1
    routes.reverse()
    return routes


def dp_split_capacity(perm: List[int], n_vehicles: int, M: np.ndarray,
                      demands: List[int], capacity: int) -> List[List[int]]:
    """
    Capacity-aware DP split:
    partition the permutation into ≤ n_vehicles segments, forbidding any segment
    whose total demand exceeds capacity. Infeasible segments are treated as INF.
    """
    n = len(perm)
    if n == 0:
        return []
    for idx in perm:
        if demands[idx - 1] > capacity:
            raise ValueError(
                f"Customer {idx} demand={demands[idx - 1]} exceeds capacity={capacity}"
            )
    # precompute feasible segment costs; INF if the segment overloads capacity
    seg = [[INF] * n for _ in range(n)]
    for i in range(n):
        load = 0
        s = M[0, perm[i]]
        for j in range(i, n):
            load += demands[perm[j] - 1]  # customers are 1-based in routes
            if load > capacity:
                break  # further j will only increase load → infeasible
            if j > i:
                s += M[perm[j - 1], perm[j]]
            seg[i][j] = s + M[perm[j], 0]

    # dp[v][j] = min cost to cover first j customers using v routes
    dp = [[INF] * (n + 1) for _ in range(n_vehicles + 1)]
    prv = [[-1] * (n + 1) for _ in range(n_vehicles + 1)]
    dp[0][0] = 0.0

    for v in range(1, n_vehicles + 1):
        for j in range(v, n + 1):  # at least 1 customer per used route
            best, arg = INF, -1
            for i in range(v - 1, j):
                c = seg[i][j - 1]
                if c >= INF:
                    continue  # infeasible segment w.r.t. capacity
                cand = dp[v - 1][i] + c
                if cand < best:
                    best, arg = cand, i
            dp[v][j], prv[v][j] = best, arg

    best_v, best_cost = -1, INF
    for v in range(1, n_vehicles + 1):
        if dp[v][n] < best_cost:
            best_v, best_cost = v, dp[v][n]

    if best_v == -1:
        # No feasible partition under capacity — fail loudly.
        raise ValueError("No feasible split under capacity constraints")

    # reconstruct routes
    routes: List[List[int]] = []
    j = n
    v = best_v
    while v > 0:
        i = prv[v][j]
        routes.append(perm[i:j])
        j, v = i, v - 1
    routes.reverse()
    return routes


def split_dispatch(perm: List[int],
                   n_vehicles: int,
                   M: np.ndarray,
                   demands: List[int] | None = None,
                   capacity: int | None = None) -> List[List[int]]:
    """
    Dispatcher that selects the split method based on SPLIT_METHOD.
    - "equal":     equal_split (no capacity)
    - "dp":        dp_optimal_split (no capacity)
    - "capacity":  dp_split_capacity (requires demands & capacity)
    """
    if SPLIT_METHOD == "equal":
        return equal_split(perm, n_vehicles)
    elif SPLIT_METHOD == "dp":
        return dp_optimal_split(perm, n_vehicles, M)
    elif SPLIT_METHOD == "capacity":
        if demands is None or capacity is None:
            raise ValueError("split_dispatch(capacity): 'demands' and 'capacity' are required")
        return dp_split_capacity(perm, n_vehicles, M, demands, capacity)
    else:
        raise ValueError(f"Unknown SPLIT_METHOD: {SPLIT_METHOD}")
