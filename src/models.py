from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict

SizeCategory = Literal['small', 'medium', 'large']
Route = List[int]
Solution = List[Route]


@dataclass
class VRPInstance:
    name: str
    category: SizeCategory
    depot: Tuple[float, float]  # depot
    customers: List[Tuple[float, float]]  # a set of customers with known locations
    n_vehicles: int  # a fleet of vehicles with limited capacity
    meta: Dict[str, str]
