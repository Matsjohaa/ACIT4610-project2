
# Multi-Objective CVRP (NSGA-II / VEGA)

This project has been refactored from a single-objective GA for VRP into a multi-objective evolutionary framework for the Capacitated Vehicle Routing Problem (CVRP).

## Problem & Objectives
We solve a small CVRP demo instance with:
- Vehicle capacity Q = 10
- Customer demands = [3, 4, 7, 2, 5]
- Up to 3 vehicles (feasible example decomposition):
	- Route 1: 1 (3) → 2 (4)
	- Route 2: 3 (7)
	- Route 3: 4 (2) → 5 (5)

Objectives (both minimized):
1. Total distance of all routes.
2. Route distance imbalance measured as the standard deviation of per-route distances (encourages similar route lengths).

Infeasible permutations (capacity violation after optimal DP split) receive large penalty values (1e9, 1e9).

## Algorithms Implemented
Two MOEAs selectable via `MOEA_ALGORITHM` in `src/constants.py`:
- `NSGA2`: Fast non-dominated sorting + crowding distance.
- `VEGA`: Vector Evaluated Genetic Algorithm (objective-specialized selection groups).

Both use permutation-based representation with k-parent (folded) crossover (OX | PMX | ERX) and standard mutations (swap / inversion / insert).

## Key Files
```text
src/
	constants.py   # MOEA parameters, demo instance data, algorithm choice
	moea.py        # Individual class, evaluation, NSGA-II + VEGA loops
	split.py       # DP capacity-aware splitting (retained)
	crossover.py   # Crossover operators
	mutation.py    # Mutation operators
	distances.py   # Euclidean distances & matrix utilities
	run_check.py   # Entry point: runs chosen MOEA and prints Pareto front
```

Removed legacy single-objective modules (`ga.py`, `fitness.py`, `experiments.py`, `metrics.py`).

## Running
```bash
pip install -r requirements.txt
python run_check.py
```
Output: a list of non-dominated solutions (permutations) with (total_distance, std_dev) and decoded routes.

Minimal footprint: legacy GA experiment/data/visualization modules and generated CSV results were removed for a lean MOEA core. Only `numpy` remains as a dependency.

To switch algorithm:
Edit `MOEA_ALGORITHM` in `src/constants.py` to either `"NSGA2"` or `"VEGA"`.

Enable Pareto Plot:
Set `ENABLE_PLOT = True` in `src/constants.py` (requires `matplotlib`). A scatter of (total distance vs std dev) will appear after the run.
The plot PNG is saved to `results/pareto_<algorithm>.png`.

## Extending
- Replace the embedded demo instance by loading larger instances (reintroduce JSON + instance loader if needed).
- Add more objectives (e.g., number of vehicles used, max route length) by extending `evaluate_individual` in `moea.py`.
- Integrate visualization by adapting previous plotting utilities to multi-objective scatter plots.

## License
Educational / academic use example (no formal license text supplied).
