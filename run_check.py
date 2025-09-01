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
from src.experiments import run_experiments
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "instances"
RESULTS = ROOT / "results"


def summarize_routes_legs(routes, M):
    """
    Return (legs, total_distance) where:
      legs = list of strings like "u->v: d.dd"
      total_distance = float
    Only uses final routes passed in; NO per-iteration prints.
    """
    legs = []
    total = 0.0
    for r in routes:
        if not r:
            continue
        prev = 0  # depot
        for node in r:
            d = M[prev, node]
            legs.append(f"{prev}->{node}: {d:.2f}")
            total += d
            prev = node
        d = M[prev, 0]  # back to depot
        legs.append(f"{prev}->0: {d:.2f}")
        total += d
    return legs, total


def print_routes_table(routes, M, title="Routes with legs"):
    """
    Print each route as a table: From | To | Distance.
    Also prints per-route and total distance.
    """
    print(f"\n=== {title} ===")
    total = 0.0
    for ridx, r in enumerate(routes, start=1):
        if not r:
            continue
        print(f"\nRoute #{ridx}: {r}")
        print("  From | To | Dist")
        print("  -----+----+------")
        prev = 0  # depot
        route_dist = 0.0
        for node in r:
            d = M[prev, node]
            print(f"  {prev:>4} | {node:<2} | {d:>5.2f}")
            route_dist += d
            prev = node
        back = M[prev, 0]
        print(f"  {prev:>4} | {0:<2} | {back:>5.2f}")
        route_dist += back
        print(f"  Route distance = {route_dist:.2f}")
        total += route_dist
    print(f"\nTOTAL distance = {total:.2f}")
    return total


def print_legs_block(title, legs, total):
    print(f"\n--- {title} (legs) ---")
    print(", ".join(legs))
    print(f"Total: {total:.2f}")


# ---------- pipeline entry points ----------

def demo_splits() -> None:
    """
    Show baseline distances for a trivial permutation (final-only leg breakdowns):
    - equal_split (uniform chunks)
    - dp_optimal_split (no capacity)
    - dp_split_capacity (capacity-feasible)
    """
    inst = load_instance(CURRENT_INSTANCE)
    M = distance_matrix(inst.depot, inst.customers)
    validate_instance(M, len(inst.customers))

    perm = list(range(1, len(inst.customers) + 1))
    demands = [c.demand for c in inst.customers]

    # equal_split (табличный вывод)
    routes_eq = equal_split(perm, inst.n_vehicles)
    d_equal = print_routes_table(routes_eq, M, "equal_split - naive")

    # dp_optimal_split (no cap)
    routes_nocap = dp_optimal_split(perm, inst.n_vehicles, M)
    d_dp = print_routes_table(routes_nocap, M, "dp_optimal_split (no cap)")

    # dp_split_capacity (cap)
    routes_cap = dp_split_capacity(perm, inst.n_vehicles, M, demands, inst.capacity)
    d_dp_cap = print_routes_table(routes_cap, M, "dp_split_capacity (cap)")

    # Summary table (как раньше)
    print_table(
        header=["Split", "Total distance"],
        rows=[
            ["equal_split - naive", f"{d_equal:.2f}"],
            ["dp_optimal_split (no cap) - exact DP ignoring capacity (can overload vehicles)", f"{d_dp:.2f}"],
            ["dp_split_capacity (cap) - exact DP with capacity constraint", f"{d_dp_cap:.2f}"],
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

    best_dist_check = print_routes_table(routes_ga, M, "GA Best (cap)")

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


# if __name__ == "__main__":
#     """
#     Entry point:
#     1) Make sure instances exist and are readable
#     2) Print baseline split distances
#     3) Run GA, print metrics as a small table, visualize, and save CSVs
#     """
#     ensure_instances()
#     demo_splits()
#     demo_ga_with_metrics()
#     print_rule()
#     print("OK.")


if __name__ == "__main__":
    ensure_instances()
    demo_splits()
    demo_ga_with_metrics()
    print_rule()
    print("Running full experiment batch...")
    results = run_experiments()
    pd.DataFrame(results).to_csv("experiment_results.csv", index=False)
    print("Saved experiment_results.csv")
