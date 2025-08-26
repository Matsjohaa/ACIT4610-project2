import numpy as np
from .models import Solution


def validate_instance(M: np.ndarray, n_customers: int):
    assert M.shape[0] == M.shape[1] == (n_customers + 1), 'Bad matrix size'
    assert np.allclose(np.diag(M), 0.0), 'Diagonal must be zero'
    assert (M >= 0).all(), 'Distances must be non-negative'


def validate_solution(sol: Solution, n_customers: int, n_vehicles: int):
    assert len(sol) <= n_vehicles, 'Too many routes'
    all_visits = [c for r in sol for c in r]
    assert sorted(all_visits) == list(range(1, n_customers + 1)), 'Each customer must be visited exactly once'
