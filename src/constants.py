from pathlib import Path

# ============================================================
# Multi-Objective Evolutionary Algorithm (MOEA) configuration
# ============================================================

# Instance selection: set to a specific instance (e.g., "small_01") or "ALL" to iterate.
INSTANCE_NAME: str = "large_02"

# Active algorithm: options: "NSGA2", "VEGA"
MOEA_ALGORITHM: str = "NSGA2"

# Population size & generations (shared for algorithms)
POP_SIZE: int = 80
GENERATIONS: int = 150

# Variation parameters
PARENTS_K: int = 2               # still using permutation-based crossover
CROSSOVER_METHOD: str = "OX"     # OX | PMX | ERX | mixed 
PC: float = 0.9                  # crossover probability
PM: float = 0.1                  # mutation probability
MUTATION_METHOD: str = "swap"    # swap | inversion | insert


# ============================================================
# Split method, capacity is default and recommended
# equal | dp | capacity 
# ============================================================
SPLIT_METHOD: str = "capacity"

# Root paths retained for potential future expansion (instances folder may be unused now)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "instances"

# Optional plotting toggle
ENABLE_PLOT: bool = True

# ============================================================
# Advanced tuning parameters (added for improved Pareto quality)
# ============================================================
# Probability to apply local search (e.g., 2-opt) to a child
LOCAL_SEARCH_PROB: float = 0.4

# Adaptive mutation toggle and floor value. If enabled, PM is scaled over generations
ADAPTIVE_PM: bool = True
PM_FLOOR: float = 0.02

# Allow mixing multiple crossover operators; if CROSSOVER_METHOD == "mixed" we randomly pick.
MIXED_CROSSOVERS = ["OX", "PMX", "ERX"]

# (Removed unused soft-penalty coefficient and balance mode toggle for simplicity)


