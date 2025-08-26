# Decode a permutation into ≤ n_vehicles routes.
from typing import List
import numpy as np


def equal_split(perm: List[int], n_vehicles: int):
    n = len(perm)
    per = max(1, (n + n_vehicles - 1) // n_vehicles)
    routes = [perm[i:i + per] for i in range(0, n, per)]
    return routes[:n_vehicles]


def single_route_cost(perm, i, j, M):
    # perm[i..j] inclusive
    if i > j: return 0.0
    s = M[0, perm[i]]
    for t in range(i, j):
        s += M[perm[t], perm[t + 1]]
    s += M[perm[j], 0]
    return s


def dp_optimal_split(perm: List[int], n_vehicles: int, M: np.ndarray):
    # Dynamic programming to minimize sum of route costs over ≤ n_vehicles segments
    n = len(perm)
    if n == 0: return []
    # precompute segment costs
    seg = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            seg[i][j] = single_route_cost(perm, i, j, M)
    INF = 1e100
    dp = [[INF] * (n + 1) for _ in range(n_vehicles + 1)]
    prv = [[-1] * (n + 1) for _ in range(n_vehicles + 1)]
    dp[0][0] = 0.0
    for v in range(1, n_vehicles + 1):
        for j in range(1, n + 1):
            best, arg = INF, -1
            for i in range(v - 1, j):
                cand = dp[v - 1][i] + seg[i][j - 1]
                if cand < best:
                    best, arg = cand, i
            dp[v][j], prv[v][j] = best, arg
    # pick best v (1..n_vehicles)
    best_v, best_cost = 1, dp[1][n]
    for v in range(2, n_vehicles + 1):
        if dp[v][n] < best_cost:
            best_v, best_cost = v, dp[v][n]
    # reconstruct routes
    routes = []
    j = n
    v = best_v
    while v > 0 and j > 0:
        i = prv[v][j]
        routes.append(perm[i:j])
        j = i
        v -= 1
    routes.reverse()
    return routes
