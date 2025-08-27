import numpy as np

from .constants import CURRENT_INSTANCE
from .io_utils import load_instance
from .models import Solution
from pathlib import Path
from src.data_gen import make_default_instances

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'instances'
RESULTS = ROOT / 'results'


def validate_instance(M: np.ndarray, n_customers: int):
    assert M.shape[0] == M.shape[1] == (n_customers + 1), 'Bad matrix size'
    assert np.allclose(np.diag(M), 0.0), 'Diagonal must be zero'
    assert (M >= 0).all(), 'Distances must be non-negative'


def validate_solution(sol: Solution, n_customers: int, n_vehicles: int):
    assert len(sol) <= n_vehicles, 'Too many routes'
    all_visits = [c for r in sol for c in r]
    assert sorted(all_visits) == list(range(1, n_customers + 1)), 'Each customer must be visited exactly once'


def ensure_instances():
    from json import JSONDecodeError
    if not any(DATA_DIR.glob('*.json')):
        print('Generate instances...')
        make_default_instances(DATA_DIR)
        return
    try:
        _ = load_instance(CURRENT_INSTANCE)
    except (JSONDecodeError, ValueError, FileNotFoundError) as e:
        print('Instance looks corrupted, regenerating...', e)
        for p in DATA_DIR.glob('*.json'):
            p.unlink(missing_ok=True)
        make_default_instances(DATA_DIR)


def validate_capacity(routes: list[list[int]], demands: list[int], capacity: int) -> None:
    for k, r in enumerate(routes, start=1):
        load = sum(demands[i-1] for i in r)
        assert load <= capacity, f"Route {k} overload: {load} > {capacity}"