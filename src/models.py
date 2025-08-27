from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict

SizeCategory = Literal['small', 'medium', 'large']
Route = List[int]
Solution = List[Route]


@dataclass
class Customer:
    x: float
    y: float
    demand: int


@dataclass
class VRPInstance:
    name: str
    category: SizeCategory
    depot: tuple[float, float]
    customers: list[Customer]
    n_vehicles: int
    capacity: int
    meta: dict[str, str]