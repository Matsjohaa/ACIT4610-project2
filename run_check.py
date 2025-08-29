from pathlib import Path
import random
from src.io_utils import load_instance
from src.distances import distance_matrix, route_length
from src.validate import validate_instance, ensure_instances, validate_capacity
from src.split import equal_split, dp_optimal_split, dp_split_capacity
from src.fitness import total_distance_from_perm, fitness_capacity_feasible
from src.ga import evolve
from src.constants import GA_PRESETS, GA_ACTIVE_PRESET, CURRENT_INSTANCE
from src.visualize import compare_solutions, plot_convergence, print_rule, print_kv, print_table
from src.metrics import GAMetrics, save_convergence_csv, save_metrics_csv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "instances"
RESULTS = ROOT / "results"


# ---------- pipeline entry points ----------

def demo_splits() -> None:
    """
    Show baseline distances for a trivial permutation using two split strategies:
    - equal_split (uniform chunks)
    - dp_optimal_split (DP, capacity-agnostic)
    """
    inst = load_instance(CURRENT_INSTANCE)
    M = distance_matrix(inst.depot, inst.customers)
    validate_instance(M, len(inst.customers))

    perm = list(range(1, len(inst.customers) + 1))
    d_equal = total_distance_from_perm(perm, inst.n_vehicles, M, equal_split)
    d_dp = total_distance_from_perm(perm, inst.n_vehicles, M, dp_optimal_split)

    # NEW: capacity-feasible baseline on the same trivial permutation
    demands = [c.demand for c in inst.customers]
    routes_cap = dp_split_capacity(perm, inst.n_vehicles, M, demands, inst.capacity)
    d_dp_cap = sum(route_length(r, M) for r in routes_cap)

    print_table(
        header=["Split", "Total distance"],
        rows=[
            ["equal_split - naive", f"{d_equal:.2f}"],
            ["dp_optimal_split (no cap) - exact DP ignoring capacity (can overload vehicles)", f"{d_dp:.2f}"],
            ["dp_split_capacity (cap) - exact DP with capacity constraint", f"{d_dp_cap:.2f}"],  # compare GA vs this
        ],
        widths=[28, 16],
    )


def demo_ga_with_metrics() -> None:
    """
    Run GA with a capacity-feasible fitness, print metrics in a compact table,
    visualize baseline vs GA, plot convergence, and save CSV summaries.
    """
    rng = random.Random(42)
    preset_name = GA_ACTIVE_PRESET
    params = GA_PRESETS[preset_name]

    inst = load_instance(CURRENT_INSTANCE)
    M = distance_matrix(inst.depot, inst.customers)
    validate_instance(M, len(inst.customers))

    # GA run (capacity-feasible fitness)
    best_ind, best_fit, info = evolve(
        M, inst.n_vehicles,
        pop_size=params["pop_size"],
        generations=params["generations"],
        pc=params["pc"],
        pm=params["pm"],
        rng=rng,
        fitness_fn=lambda perm, n_vehicles, M: fitness_capacity_feasible(perm, inst, M),
        record_history=True,
    )

    # Decode best permutation with capacity-aware split
    demands = [c.demand for c in inst.customers]
    total_dem = sum(demands)
    routes_ga = dp_split_capacity(best_ind, inst.n_vehicles, M, demands, inst.capacity)
    validate_capacity(routes_ga, demands, inst.capacity)
    loads = [sum(demands[i - 1] for i in r) for r in routes_ga]
    used = sum(1 for r in routes_ga if r)
    best_dist = sum(route_length(r, M) for r in routes_ga)

    # Baseline: DP split on identity permutation
    identity_perm = list(range(1, len(inst.customers) + 1))
    # Baseline A: DP split (no capacity)
    routes_nocap = dp_optimal_split(identity_perm, inst.n_vehicles, M)
    dist_nocap = sum(route_length(r, M) for r in routes_nocap)
    used_nocap = sum(1 for r in routes_nocap if r)

    print_kv("DP split (no cap) distance", f"{dist_nocap:.2f}")
    print_kv("DP split (no cap) vehicles used", f"{used_nocap}/{inst.n_vehicles}")

    # Baseline B: DP split (with capacity)
    routes_cap = dp_split_capacity(identity_perm, inst.n_vehicles, M, demands, inst.capacity)
    dist_cap = sum(route_length(r, M) for r in routes_cap)
    used_cap = sum(1 for r in routes_cap if r)

    print_kv("DP split (cap) distance", f"{dist_cap:.2f}")
    print_kv("DP split (cap) vehicles used", f"{used_cap}/{inst.n_vehicles}")

    # Improvement of GA compared to DP with cap
    improvement_pct = 100 * (dist_cap - best_dist) / dist_cap
    print_kv("Improvement over DP split (cap)", f"{improvement_pct:.2f}%")

    # Fancy console output
    print_rule()
    print_kv("GA preset", f"{preset_name} -> {params}")
    print_kv("Capacity per vehicle", inst.capacity)
    print_kv("Total demand", f"{total_dem}  |  routes used {used}/{inst.n_vehicles}")
    print_table(
        header=["Route #", "Load", "Len (nodes)"],
        rows=[[str(i + 1), str(loads[i]), str(len(routes_ga[i]))] for i in range(len(routes_ga))],
        widths=[8, 8, 12],
    )
    print_table(
        header=["Metric", "Value"],
        rows=[
            ["best_fitness", f"{best_fit:.6f}"],
            ["best_distance", f"{best_dist:.2f}"],
            ["runtime_s", f"{info['runtime_s']:.3f}"],
            ["evaluations", f"{info['evaluations']}"],
        ],
        widths=[16, 16],
    )

    min_segs_ga = min_segments_for_perm(best_ind, demands, inst.capacity)
    print_kv("Min segments for GA best", f"{min_segs_ga}/{inst.n_vehicles}")

    identity_perm = list(range(1, len(inst.customers) + 1))
    routes_baseline = dp_split_capacity(identity_perm, inst.n_vehicles, M, demands, inst.capacity)

    # Visualize
    compare_solutions(
        inst,
        routes_a=routes_baseline,
        routes_b=routes_ga,
        title_a="DP split (capacity-feasible, identity order)",
        title_b="GA Best (capacity-feasible)",
    )
    plot_convergence(info["fitness_history"], title=f"Convergence ({preset_name})")

    # Save CSV outputs
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_convergence_csv(RESULTS / f"convergence_{inst.name}_{preset_name}.csv", info["fitness_history"])
    metrics_row = GAMetrics(
        preset=preset_name,
        instance=inst.name,
        n_customers=len(inst.customers),
        n_vehicles=inst.n_vehicles,
        best_fitness=best_fit,
        best_distance=best_dist,
        runtime_s=info["runtime_s"],
        generations=params["generations"],
        pop_size=params["pop_size"],
        evaluations=info["evaluations"],
        evals_per_sec=info["evaluations"] / info["runtime_s"] if info["runtime_s"] > 0 else float("nan"),
    )
    save_metrics_csv(RESULTS / f"metrics_{inst.name}.csv", [metrics_row])

def min_segments_for_perm(perm, demands, capacity):
    n = len(perm)
    INF = 10 ** 9
    dp = [INF] * (n + 1)
    dp[0] = 0
    for j in range(1, n + 1):
        s = 0
        for i in range(j, 0, -1):
            s += demands[perm[i - 1] - 1]
            if s > capacity:
                break
            dp[j] = min(dp[j], dp[i - 1] + 1)
    return dp[n]


if __name__ == "__main__":
    """
    Entry point:
    1) Make sure instances exist and are readable
    2) Print baseline split distances
    3) Run GA, print metrics as a small table, visualize, and save CSVs
    """
    ensure_instances()
    demo_splits()
    demo_ga_with_metrics()
    print_rule()
    print("OK.")
