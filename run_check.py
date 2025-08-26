from pathlib import Path
import random
from src.io_utils import load_instance
from src.data_gen import make_default_instances
from src.distances import distance_matrix
from src.validate import validate_instance
from src.split import equal_split, dp_optimal_split
from src.fitness import total_distance_from_perm, fitness_maximizing
from src.ga import evolve
from src.constants import GA_PRESETS

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'instances'


def ensure_instances():
    if not any(DATA_DIR.glob('*.json')):
        print('Generate 6 instances...')
        make_default_instances(DATA_DIR)


def demo_splits():
    inst_path = DATA_DIR / 'small_01.json'
    inst = load_instance(inst_path)
    M = distance_matrix(inst.depot, inst.customers)
    validate_instance(M, len(inst.customers))

    perm = list(range(1, len(inst.customers) + 1))  # простая пермутация
    d_equal = total_distance_from_perm(perm, inst.n_vehicles, M, equal_split)
    d_dp = total_distance_from_perm(perm, inst.n_vehicles, M, dp_optimal_split)
    print(f"[{inst.name}] vehicles={inst.n_vehicles} customers={len(inst.customers)}")
    print("  distance (equal split):", round(d_equal, 2))
    print("  distance (DP split):   ", round(d_dp, 2))


def demo_ga():
    # Мини-демо GA на small_01 с пресетом 'fast' (быстро и недорого)
    rng = random.Random(42)
    inst_path = DATA_DIR / 'small_01.json'
    inst = load_instance(inst_path)
    M = distance_matrix(inst.depot, inst.customers)
    validate_instance(M, len(inst.customers))

    params = GA_PRESETS['fast']
    best_ind, best_fit = evolve(
        M, inst.n_vehicles,
        pop_size=params['pop_size'],
        generations=params['generations'],
        pc=params['pc'],
        pm=params['pm'],
        rng=rng
    )
    print("\nGA demo (fast preset):")
    print("  best_fitness =", round(best_fit, 6))


if __name__ == '__main__':
    ensure_instances()
    demo_splits()
    demo_ga()
    print("\nOK.")
