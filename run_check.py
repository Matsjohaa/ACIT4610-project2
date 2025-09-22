from pathlib import Path
from src.moea import run_moea, Individual
from src.constants import MOEA_ALGORITHM, ENABLE_PLOT, INSTANCE_NAME
from src.instances import load_instance, list_instances, InstanceData
from src.split import dp_split_capacity
from src.distances import route_length
import math
import time

ROOT = Path(__file__).resolve().parent


def _summarize_routes(ind: Individual, M, inst: InstanceData):
    try:
        routes = dp_split_capacity(ind.perm, inst.n_vehicles, M, inst.demands, inst.capacity)
    except ValueError:
        return [], math.inf, math.inf
    dists = [route_length(r, M) for r in routes if r]
    total = sum(dists)
    if len(dists) <= 1:
        std_dev = 0.0
    else:
        mean = sum(dists)/len(dists)
        var = sum((x-mean)**2 for x in dists)/len(dists)
        std_dev = math.sqrt(var)
    return routes, total, std_dev


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

def print_pareto(pareto, M, inst: InstanceData):
    print(f"Instance: {inst.name} | Algorithm: {MOEA_ALGORITHM}")
    print("Pareto front (distance, std_dev):")
    seen = set()
    for i, ind in enumerate(sorted(pareto, key=lambda x: x.objectives)):
        key = ind.objectives
        if key in seen:
            continue
        seen.add(key)
        routes, total, std_dev = _summarize_routes(ind, M, inst)
        print(f"#{len(seen):02d} perm={ind.perm} dist={total:.2f} std={std_dev:.2f} routes={routes}")


if __name__ == "__main__":
    start_time = time.time()
    names = list_instances() if INSTANCE_NAME.upper() == 'ALL' else [INSTANCE_NAME]
    for name in names:
        inst = load_instance(name)
        pareto, M, inst = run_moea(seed=42, instance=inst)
        print_pareto(pareto, M, inst)
        if ENABLE_PLOT:
            try:
                from src.plot_utils import plot_pareto, plot_routes, plot_route_comparison
                pareto_path = plot_pareto(pareto, inst, show=False, out_dir="results")
                print(f"Saved Pareto plot to {pareto_path}")
                best_dist = min(pareto, key=lambda ind: ind.objectives[0])
                best_std = min(pareto, key=lambda ind: ind.objectives[1])
                comp_path = plot_route_comparison(best_dist, best_std, inst, M, show=False, out_dir="results")
                print(f"Saved comparison route plot to {comp_path}")
            except Exception as e:
                print(f"Plotting failed: {e}")
        print("---")
    end_time = time.time()
    runtime = end_time - start_time
    print("Finished.")
    print(f"Total runtime: {runtime:.2f} seconds")


# if __name__ == "__main__":
#     ensure_instances()
#     demo_splits()
#     demo_ga_with_metrics()
#     print_rule()
#     print("Running full experiment batch...")
#     results = run_experiments()
#     pd.DataFrame(results).to_csv("experiment_results.csv", index=False)
#     print("Saved experiment_results.csv")
