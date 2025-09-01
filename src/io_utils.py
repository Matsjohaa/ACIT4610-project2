import json
from pathlib import Path
import numpy as np
from .models import VRPInstance, Customer


def save_json(obj, path: Path):
    """
    Save a Python object to a JSON file with UTF-8 encoding.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    """
    Load a JSON file and return the parsed Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_instance(inst: VRPInstance, path: Path) -> None:
    """
    Convert a VRPInstance (dataclass) into a dictionary and save it as JSON.
    """
    obj = {
        "name": inst.name,
        "category": inst.category,
        "depot": list(inst.depot),  # store tuple as list for JSON
        "customers": [
            {"x": c.x, "y": c.y, "demand": c.demand} for c in inst.customers
        ],
        "n_vehicles": inst.n_vehicles,
        "capacity": inst.capacity,
        "meta": inst.meta,
    }
    save_json(obj, path)


def load_instance(path: Path, default_capacity: int | None = None) -> VRPInstance:
    """
    Load a VRPInstance from a JSON file and reconstruct dataclasses.
    """
    d = load_json(path)

    if "capacity" not in d:
        if default_capacity is None:
            raise ValueError("capacity missing in instance JSON")
        d["capacity"] = default_capacity

    # reconstruct customers as Customer dataclasses
    customers = []
    for c in d["customers"]:
        if isinstance(c, dict):
            customers.append(Customer(**c))
        elif isinstance(c, (list, tuple)) and len(c) == 2:
            # legacy format: [x, y] with no demand → demand = 1
            customers.append(Customer(x=c[0], y=c[1], demand=1))
        else:
            raise ValueError(f"Unexpected customer format: {c}")
    d["customers"] = customers

    # ensure depot is always a tuple
    d["depot"] = tuple(d["depot"])

    return VRPInstance(**d)


def save_dist_matrix(M, path: Path) -> None:
    """
    Save a NumPy distance matrix to a .npy file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), M)


def load_dist_matrix(path: Path):
    """
    Load a NumPy distance matrix from a .npy file.
    """
    return np.load(str(path))
