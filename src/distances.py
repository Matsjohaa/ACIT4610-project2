import math
from typing import Tuple
import numpy as np
from .models import Route, Solution, Customer

Point = Tuple[float, float]


def euclid(a: Point, b: Point) -> float:
    """
    Compute Euclidean distance between two points (x,y).
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_matrix(depot: tuple[float, float], customers: list[Customer]) -> np.ndarray:
    """
    Build a full symmetric distance matrix for depot + customers.
    Index 0 corresponds to the depot, indices 1..N correspond to customers.
    """
    pts = [depot] + [(c.x, c.y) for c in customers]
    n = len(pts)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = euclid(pts[i], pts[j])
            M[i, j] = M[j, i] = d
    return M


def route_length(route: Route, M: np.ndarray) -> float:
    """
    Compute the length of a single route:
    depot -> first customer -> ... -> last customer -> depot.
    """
    if not route:
        return 0.0
    s = M[0, route[0]]
    for a, b in zip(route, route[1:]):
        s += M[a, b]
    s += M[route[-1], 0]
    return s


def total_length(solution: Solution, M: np.ndarray) -> float:
    """
    Compute the total length of a full VRP solution
    (sum of all individual route lengths).
    """
    return sum(route_length(r, M) for r in solution)
