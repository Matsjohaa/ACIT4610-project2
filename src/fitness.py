import numpy as np
from typing import List, Callable
from .distances import route_length
from .split import equal_split, dp_optimal_split


def total_distance_from_perm(perm: List[int], n_vehicles: int, M: np.ndarray,
                             split_fn: Callable = equal_split) -> float:
    routes = split_fn(perm, n_vehicles, M) if split_fn is dp_optimal_split else split_fn(perm, n_vehicles)
    return sum(route_length(r, M) for r in routes)


def fitness_maximizing(perm: List[int], n_vehicles: int, M: np.ndarray,
                       split_fn: Callable = equal_split) -> float:
    d = total_distance_from_perm(perm, n_vehicles, M, split_fn)
    return 1.0 / (1.0 + d)


def fitness_minimizing(perm: List[int], n_vehicles: int, M: np.ndarray,
                       split_fn: Callable = equal_split) -> float:
    return total_distance_from_perm(perm, n_vehicles, M, split_fn)
