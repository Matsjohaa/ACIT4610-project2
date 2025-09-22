from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Dict

DATA_DIR = Path(__file__).resolve().parent / 'data' / 'instances'


@dataclass
class InstanceData:
	name: str
	capacity: int
	n_vehicles: int
	demands: List[int]
	coords: Dict[int, tuple]


def list_instances() -> List[str]:
	return [p.stem for p in DATA_DIR.glob('*.json')]


def load_instance(name: str) -> InstanceData:
	path = DATA_DIR / f"{name}.json"
	if not path.exists():
		raise FileNotFoundError(f"Instance file not found: {path}")
	with open(path, 'r') as f:
		raw_lines = f.readlines()
	# Allow // comments at start of file or in-line
	filtered = '\n'.join(l for l in raw_lines if not l.lstrip().startswith('//'))
	data = json.loads(filtered)
	# Schema: depot: [x,y]; customers: list of {x,y,demand}
	depot = tuple(data['depot'])
	customers = data['customers']
	coords: Dict[int, tuple] = {0: depot}
	demands: List[int] = []
	for idx, c in enumerate(customers, start=1):
		coords[idx] = (c['x'], c['y'])
		demands.append(c['demand'])
	return InstanceData(
		name=data.get('name', name),
		capacity=data['capacity'],
		n_vehicles=data['n_vehicles'],
		demands=demands,
		coords=coords
	)

__all__ = ["InstanceData", "list_instances", "load_instance"]
