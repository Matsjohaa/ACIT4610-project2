from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict

SizeCategory = Literal['small', 'medium', 'large']
Route = List[int]
Solution = List[Route]


@dataclass
class VRPInstance:
    name: str
    category: SizeCategory
    depot: Tuple[float, float]
    customers: List[Tuple[float, float]]
    n_vehicles: int
    meta: Dict[str, str]
