
# Main constants and parameter presets

CATEGORIES = {
    'small':  {'n_customers_range': (10, 20), 'n_vehicles_range': (2, 10)},
    'medium': {'n_customers_range': (15, 30), 'n_vehicles_range': (11, 25)},
    'large':  {'n_customers_range': (20, 50), 'n_vehicles_range': (26, 50)},
}

# Coordinates diapason while generating random instances
XY_RANGE = (0.0, 100.0)

# GA parameters presets (can be adjusted)
GA_PRESETS = {
    'fast': {'pop_size': 40,  'generations': 120, 'pc': 0.9,  'pm': 0.10},
    'balanced': {'pop_size': 80,  'generations': 300, 'pc': 0.9,  'pm': 0.08},
    'thorough': {'pop_size': 120, 'generations': 600, 'pc': 0.95, 'pm': 0.05},
}

# 6 instances specification (Name, Category, Seed, Client amount, Car amount)
INSTANCE_SPECS = [
    ('small_01',  'small',  13,  12,  4),
    ('small_02',  'small',  18,  18,  7),
    ('medium_01', 'medium', 22,  20, 14),
    ('medium_02', 'medium', 28,  26, 20),
    ('large_01',  'large',  35,  30, 30),
    ('large_02',  'large',  48,  45, 45),
]
