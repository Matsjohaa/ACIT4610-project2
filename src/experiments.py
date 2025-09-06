import time
import numpy as np
import random

from src.constants import INSTANCE_SPECS, GA_PRESETS
from src.fitness import fitness_capacity_feasible
from src.ga import evolve
from pathlib import Path

from src.io_utils import load_instance
from src.distances import distance_matrix, route_length
from src.split import equal_split, dp_optimal_split, dp_split_capacity

N_RUNS = 10  # run each experiment multiple times
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "instances"


def run_experiments():
    results = []

    for inst_name, category, n_customers, n_vehicles, cap in INSTANCE_SPECS:
        inst_path = DATA_DIR / f"{inst_name}.json"
        inst = load_instance(inst_path)
        M = distance_matrix(inst.depot, inst.customers)

        # --- DP baseline with capacity (on identity order) ---
        perm_identity = list(range(1, len(inst.customers) + 1))
        demands = [c.demand for c in inst.customers]
        routes_cap = dp_split_capacity(perm_identity, inst.n_vehicles, M, demands, inst.capacity)
        dist_cap = sum(route_length(r, M) for r in routes_cap)

        for preset_name, params in GA_PRESETS.items():
            run_dists = []
            run_times = []
            run_improvements = []

            for run in range(N_RUNS):
                rng = random.Random(run)  # reproducible per run
                start = time.time()

                best_ind, best_fit, info = evolve(
                    M, inst.n_vehicles,
                    pop_size=params["pop_size"],
                    generations=params["generations"],
                    pc=params["pc"],
                    pm=params["pm"],
                    rng=rng,
                    fitness_fn=lambda perm, n_vehicles, M: fitness_capacity_feasible(perm, inst, M),
                    record_history=False,
                )

                duration = time.time() - start
                best_dist = 1.0 / best_fit - 1.0  # back-transform fitness to distance

                # compute % improvement vs baseline DP (cap)
                improvement_pct = 100 * (dist_cap - best_dist) / dist_cap

                run_dists.append(best_dist)
                run_times.append(duration)
                run_improvements.append(improvement_pct)

            results.append({
                "instance": inst_name,
                "category": category,
                "preset": preset_name,
                "best": np.min(run_dists),
                "avg": np.mean(run_dists),
                "worst": np.max(run_dists),
                "time_avg": np.mean(run_times),
                "improvement_avg": np.mean(run_improvements),  # new column
                "dp_cap_baseline": dist_cap,                   # optional for reference
            })

    return results
