import random
from pathlib import Path
from typing import Tuple
from .models import VRPInstance, SizeCategory, Customer
from .constants import XY_RANGE, INSTANCE_SPECS, DEMAND_RANGE, TARGET_UTIL
import math


def gen_instance(
    name: str,
    category: SizeCategory,
    seed: int,
    n_customers: int,
    n_vehicles: int,
    xy_range: Tuple[float, float] = XY_RANGE,
) -> VRPInstance:
    """
    Generate a single VRP instance:
    - depot placed randomly within xy_range
    - customers with random coordinates and demand in DEMAND_RANGE
    - vehicle capacity chosen as max(target-utilization based, feasibility-based, max-customer demand)
      to guarantee feasibility of the instance
    """
    rng = random.Random(seed)

    # Random depot location
    depot = (rng.uniform(*xy_range), rng.uniform(*xy_range))

    # Generate customers with random coordinates and demand
    customers = [
        Customer(
            x=rng.uniform(*xy_range),
            y=rng.uniform(*xy_range),
            demand=rng.randint(*DEMAND_RANGE)
        )
        for _ in range(n_customers)
    ]

    # Compute total demand and derive capacity per vehicle
    total_demand = sum(c.demand for c in customers)
    cap_target = int(round(total_demand / max(1, int(n_vehicles * TARGET_UTIL))))
    cap_feasible = math.ceil(total_demand / n_vehicles)
    cap_min = max(c.demand for c in customers)

    capacity = max(cap_target, cap_feasible, cap_min)

    return VRPInstance(
        name=name,
        category=category,
        depot=depot,
        customers=customers,
        n_vehicles=n_vehicles,
        capacity=capacity,
        meta={'seed': str(seed)}
    )


def make_default_instances(out_dir: Path):
    """
    Generate and save the 6 default instances defined in INSTANCE_SPECS.
    Returns a list of file paths for the created JSON files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cat, seed, n_customers, n_vehicles in INSTANCE_SPECS:
        inst = gen_instance(name, cat, seed, n_customers, n_vehicles)
        path = out_dir / f"{name}.json"
        from .io_utils import save_instance
        save_instance(inst, path)
        paths.append(path)
    return paths
