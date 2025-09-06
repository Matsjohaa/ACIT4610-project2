
# VRP GA Skeleton (Full)

A complete scaffold for solving the Vehicle Routing Problem (VRP) using a Genetic Algorithm (GA).  

- **Representation**: a permutation of all customers (1..N), depot = 0 in the distance matrix.  
- **Fitness**: the sum of route lengths after splitting the permutation into ≤ `n_vehicles` routes.  
- **Splitting**: uniform (*equal split*) and optimal (*DP*).  
- **GA**: initialization, tournament selection, OX crossover, swap mutation 

## Structure
```text
vrp_ga_skeleton_full/
├─ src/
│  ├─ constants.py        # all constants and presets, customisable
│  ├─ models.py           # dataclass VRPInstance, types
│  ├─ io_utils.py         # I/O JSON/NPY
│  ├─ distances.py        # euclidean distance, matrix, lengths
│  ├─ data_gen.py         # generation of 6 instances
│  ├─ validate.py         # invariant checks
│  ├─ split.py            # equal_split and dp_optimal_split
│  ├─ fitness.py          # fitness wrappers
│  ├─ crossover.py        # OX, PMX and ERX
│  ├─ mutation.py         # swap/inversion/insert
│  └─ ga.py               # GA loop
├─ data/instances/        # JSON and .npy
├─ results/               # experiment results
├─ run_check.py           # sanity check + mini GA demo
└─ requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt
python run_check.py
```
