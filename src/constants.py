from pathlib import Path

# ============================================================
# Genetic Algorithm configuration
# ============================================================

# Number of parents in crossover (Default = 2)
PARENTS_K: int = 2

# Type of crossover used for folding multiple parents
# Options: "OX", "PMX", "ERX"
CROSSOVER_METHOD: str = "OX"

# GA parameter presets (population size, generations, probabilities)
GA_PRESETS = {
    'fast': {'pop_size': 40, 'generations': 120, 'pc': 0.9, 'pm': 0.10},
    'balanced': {'pop_size': 80, 'generations': 300, 'pc': 0.9, 'pm': 0.08},
    'thorough': {'pop_size': 120, 'generations': 600, 'pc': 0.95, 'pm': 0.05},
}

GA_ACTIVE_PRESET = "thorough"

# ============================================================
# Demand / capacity generation configuration
# ============================================================

# Range of customer demand values (uniform random)
DEMAND_RANGE = (5, 15)

# Target fleet utilization (fraction of vehicles we want to be active)
# Example: 0.7 → aim for ~70% of vehicles to be used
TARGET_UTIL = 0.7

# ============================================================
# Instance generation configuration
# ============================================================

# Customer categories (small, medium, large) with ranges
CATEGORIES = {
    'small': {'n_customers_range': (10, 20), 'n_vehicles_range': (2, 10)},
    'medium': {'n_customers_range': (15, 30), 'n_vehicles_range': (11, 25)},
    'large': {'n_customers_range': (20, 50), 'n_vehicles_range': (26, 50)},
}

# Coordinate range for randomly generating depot and customer locations
XY_RANGE = (0.0, 100.0)

# Specification of the 6 default instances:
# (Name, Category, Seed, Number of customers, Number of vehicles)
INSTANCE_SPECS = [
    ('small_01', 'small', 13, 12, 4),
    ('small_02', 'small', 18, 18, 7),
    ('medium_01', 'medium', 22, 20, 14),
    ('medium_02', 'medium', 28, 26, 20),
    ('large_01', 'large', 35, 30, 30),
    ('large_02', 'large', 48, 45, 45),
]
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "instances"
CURRENT_INSTANCE = (DATA_DIR / 'large_01.json')

# ============================================================
# Mutation configuration
# ============================================================

# Which mutation operator to use by default
# Options: "swap", "inversion", "insert"
MUTATION_METHOD: str = "inversion"

# ============================================================
# Split (decode permutation -> routes) configuration
# ============================================================

# Which split method to use by default:
# "equal"    -> equal sized chunks (baseline, capacity-agnostic)
# "dp"       -> DP split minimizing distance (capacity-agnostic, ≤ n_vehicles)
# "capacity" -> DP split enforcing per-route capacity (≤ n_vehicles)
SPLIT_METHOD: str = "capacity"
