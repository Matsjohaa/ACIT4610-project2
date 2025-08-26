import json
from pathlib import Path
import numpy as np
from .models import VRPInstance


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_instance(inst: VRPInstance, path: Path):
    obj = {
        'name': inst.name,
        'category': inst.category,
        'depot': inst.depot,
        'customers': inst.customers,
        'n_vehicles': inst.n_vehicles,
        'meta': inst.meta,
    }
    save_json(obj, path)


def load_instance(path: Path) -> VRPInstance:
    d = load_json(path)
    return VRPInstance(**d)


def save_dist_matrix(M, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), M)


def load_dist_matrix(path: Path):
    return np.load(str(path))
