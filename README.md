## Multi-Objective CVRP (NSGA-II / VEGA)

Multi-objective evolutionary optimization for the Capacitated Vehicle Routing Problem (CVRP) with NSGA-ll and VEGA

---
### Objectives
For a solution with routes R = {r1, r2, …, rk}:
- f1 = sum of Euclidean route lengths (including depot returns)
- f2 = sqrt( (1/k) * Σ (len(ri) - mean)^2 )

This encourages low total distance while keeping route workloads similar.

---
### Instance Format (JSON)
Instances live in `src/data/instances/`.

Customer indices are implicit (1..N). Index 0 is always the depot.
---
### Repository Structure (core parts)
```text
run_check.py          # Main entry: loops instances, runs MOEA, saves/prints results
src/
	constants.py        # Configuration flags & algorithm parameters
	instances.py        # JSON parsing -> InstanceData
	moea.py             # Individual, evaluation, NSGA-II & VEGA loop
	split.py            # Capacity-aware DP splitting
	distances.py        # Distance matrix + route length helpers
	crossover.py        # Crossover operators
	mutation.py         # Mutation operators + adaptive logic
	plot_utils.py       # Pareto and route comparison plotting
	data/instances/*.json
results/              # Generated plots (Pareto + comparison)
```


---
### Quick Start
```bash
## Activate virutal enviorment
python -m venv .venv
source .venv/bin/activate  # (macOS / Linux)
## Download Dependencies
pip install -r requirements.txt
## Start project
python run_check.py
```

---
### Configuration (edit `src/constants.py`)
- `INSTANCE_NAME`: Specific instance name or `ALL`
- `MOEA_ALGORITHM`: `NSGA2` | `VEGA`
- `POP_SIZE`, `GENERATIONS`: Evolution scale
- `PC`, `PM`: Crossover & mutation probabilities (mutation may adapt)
- `ADAPTIVE_PM`, `PM_FLOOR`: Adaptive mutation scheduling
- `CROSSOVER_METHOD`: `OX` | `PMX` | `ERX` | `mixed`
- `MUTATION_METHOD`: `swap` | `inversion` | `insert`
- `LOCAL_SEARCH_PROB`: Chance to apply 2‑opt per child
- `ENABLE_PLOT`: Toggle Matplotlib output
- `OBJECTIVE_BALANCE`: Currently fixed to `std` (standard deviation)

---
### Outputs
Console (per instance):
- Header with instance + algorithm
- Pareto set (objective tuple + permutation + decoded routes summary)

Plots (if `ENABLE_PLOT = True`):
- Pareto front scatter: distance vs std deviation (saved to `results/pareto_<instance>_<algo>.png`)
- Route comparison: Best-distance vs best-std individual side-by-side with route counts (`results/routes_compare_<instance>_<algo>.png`)

---
### Interpreting the Pareto Front
- Lower-left solutions: balanced and short — usually desirable.
- Near-horizontal spread: trade-offs where distance similar, balance varies.
- Near-vertical spread: trade-offs where balance similar, distance varies.
If the front is sparse, increase `POP_SIZE` or `GENERATIONS`, or raise diversity via `mixed` crossover + higher mutation / local search probability.

---
### Extending
- Add objective: modify `evaluate_individual` in `moea.py` and adjust dominance logic (tuples auto-expand).
- Different balance metric: implement coefficient of variation (CV) and toggle with `OBJECTIVE_BALANCE`.
- Soft capacity penalties: integrate `PENALTY_OVERLOAD_ALPHA` (currently not used—hard infeasible marking is applied instead).
- Improve VEGA: Add elitism or hybrid NSGA-II replacement stage.
- Export CSV: Iterate Pareto set and write objective + permutation + route breakdown (not yet implemented).


