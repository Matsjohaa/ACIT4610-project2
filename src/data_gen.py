import random
from pathlib import Path
from typing import Tuple
from .models import VRPInstance, SizeCategory
from .constants import XY_RANGE, INSTANCE_SPECS


def gen_instance(name: str, category: SizeCategory, seed: int, n_customers: int, n_vehicles: int,
                 xy_range: Tuple[float, float] = XY_RANGE,
                 ) -> VRPInstance:
    import random
    rng = random.Random(seed)
    depot = (rng.uniform(*xy_range), rng.uniform(*xy_range))
    customers = [(rng.uniform(*xy_range), rng.uniform(*xy_range)) for _ in range(n_customers)]
    return VRPInstance(
        name=name,
        category=category,
        depot=depot,
        customers=customers,
        n_vehicles=n_vehicles,
        meta={'seed': str(seed), 'xy_range': str(xy_range)},
    )


def make_default_instances(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cat, seed, n_customers, n_vehicles in INSTANCE_SPECS:
        inst = gen_instance(name, cat, seed, n_customers, n_vehicles)
        path = out_dir / f"{name}.json"
        from .io_utils import save_instance
        save_instance(inst, path)
        paths.append(path)
    return paths
