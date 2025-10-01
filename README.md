# Multi-Objective CVRP (NSGA-II / VEGA)

Multi-objective evolutionary optimization for the Capacitated Vehicle Routing Problem (CVRP) using NSGA-II and VEGA.  
Implements two conflicting objectives: minimizing total route distance and promoting fairness among routes.

---

## Objectives

Given a solution with routes \(R = \{r_1, r_2, …, r_k\}\):

- **f1** = Total distance  
(Euclidean route lengths including depot returns)

- **f2** = Route balance  
(standard deviation of route lengths, encouraging similar workloads)


---

## Instance Format

- Instances are JSON files in `src/data/instances/`.  
- Each file contains depot, customers, demands, and capacity (derived from CVRPLIB dataset).

---

## Repository Structure

```text
run_check.py               # Main entry: loops instances, runs MOEAs, saves/prints results
src/
    constants.py           # Algorithm parameters
    crossover.py           # Crossover operators
    data_transformer.py    # Collates experiment outputs
    distances.py           # Distance matrix + route helpers
    exp_runner_threads.py  # Runs VEGA/NSGA-II across presets/seeds
    instances.py           # JSON parsing
    moea.py                # NSGA-II & VEGA implementation
    mutation.py            # Mutation operators
    plot_utils.py          # Pareto and route plotting
    split.py               # Capacity-aware splitting
    data/instances/*.json
notebooks/             # Generated plots and notebooks
```
---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Run experiments to produce fronts (This part can be slow – many runs across ALL preset/instances, especially M-n151-k12 is super slow)
cd src
python exp_runner_threads.py
# As alternative, you can download all generated fronts and reports here: https://drive.google.com/drive/folders/1rip-YUq92RbrIoCUQ2KeDxSjmBj_Mt-i?usp=sharing
# In this case: 
# 1. Place folders with instance names to exp_runner_output/fronts, 
# 2. Place runs_raw.csv, runtime_summary.csv and summary_metrics.scv - to src/results

# 2. Collate fronts outputs to produce one output file (in parquet and csv formats)
cd src
python data_transformer.py

# 3. Make sure all input files are on their place, explore and visualise in Visualisation_notebook.ipynb file
cd notebooks
```