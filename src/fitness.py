import numpy as np
from typing import List, Callable
from .distances import route_length
from .models import Customer
from .split import equal_split, dp_optimal_split, dp_split_capacity


def total_distance_from_perm(perm: List[int], n_vehicles: int, M: np.ndarray,
                             split_fn: Callable = equal_split) -> float:
    """
    Decode a permutation into routes (using a split function) and
    return the total travel distance across all routes.
    """
    routes = split_fn(perm, n_vehicles, M) if split_fn is dp_optimal_split else split_fn(perm, n_vehicles)
    return sum(route_length(r, M) for r in routes)


def fitness_maximizing(perm: List[int], n_vehicles: int, M: np.ndarray,
                       split_fn: Callable = equal_split) -> float:
    """
    Fitness function for GA (maximize).
    Returns 1 / (1 + distance) so that shorter routes
    correspond to higher fitness values.
    """
    d = total_distance_from_perm(perm, n_vehicles, M, split_fn)
    return 1.0 / (1.0 + d)


def fitness_minimizing(perm: List[int], n_vehicles: int, M: np.ndarray,
                       split_fn: Callable = equal_split) -> float:
    """
    Fitness function for GA (minimize).
    Returns the raw total distance, which must be minimized.
    """
    return total_distance_from_perm(perm, n_vehicles, M, split_fn)


# def capacity_penalty(routes: list[list[int]], customers: list[Customer], capacity: int, beta: float = 1000.0) -> float:
#     """
#     Compute a penalty for capacity violations.
#     For each route, if the load (sum of customer demands) exceeds
#     the vehicle capacity, add (excess * beta) to the penalty.
#     """
#     penalty = 0.0
#     for r in routes:
#         load = sum(customers[i - 1].demand for i in r)  # clients are 1-based
#         if load > capacity:
#             penalty += beta * (load - capacity)
#     return penalty


def fitness_capacity_feasible(perm: list[int], inst, M: np.ndarray) -> float:
    """
    Capacity-feasible fitness function.
    Uses a capacity-aware DP split (dp_split_capacity),
    ensuring all routes respect capacity constraints.
    """
    demands = [c.demand for c in inst.customers]
    routes = dp_split_capacity(perm, inst.n_vehicles, M, demands, inst.capacity)
    dist = sum(route_length(r, M) for r in routes)
    return 1.0 / (1.0 + dist)  # maximize
