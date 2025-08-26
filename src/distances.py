import math
from typing import List, Tuple
import numpy as np
from .models import Route, Solution

Point = Tuple[float, float]


def euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_matrix(depot: Point, customers: List[Point]) -> np.ndarray:
    pts = [depot] + customers
    n = len(pts)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = euclid(pts[i], pts[j])
            M[i, j] = M[j, i] = d
    return M


def route_length(route: Route, M: np.ndarray) -> float:
    if not route: return 0.0
    s = M[0, route[0]]
    for a, b in zip(route, route[1:]):
        s += M[a, b]
    s += M[route[-1], 0]
    return s


def total_length(solution: Solution, M: np.ndarray) -> float:
    return sum(route_length(r, M) for r in solution)
