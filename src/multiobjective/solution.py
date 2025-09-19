"""
Multi-Objective Solution representation for VRP
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MOSolution:
    """
    Multi-Objective Solution for Capacitated VRP
    
    Represents a solution with:
    - A permutation of customers
    - Two objectives: total distance and route balance (std dev)
    - NSGA-II specific attributes: rank, crowding distance
    - Route information after decoding
    """
    
    # Core solution representation
    permutation: List[int] = field(default_factory=list)
    
    # Objectives (both to be minimized)
    objectives: Optional[Tuple[float, float]] = None  # (total_distance, std_deviation)
    
    # Decoded routes
    routes: Optional[List[List[int]]] = None
    
    # Solution quality
    feasible: bool = True
    constraint_violations: int = 0
    
    # NSGA-II specific attributes
    rank: Optional[int] = None
    crowding_distance: float = 0.0
    
    # For non-dominated sorting
    domination_count: int = 0
    dominated_solutions: List['MOSolution'] = field(default_factory=list)
    
    # Metadata
    generation: int = -1
    evaluation_time: float = 0.0
    
    def __post_init__(self):
        """Initialize after creation"""
        if not isinstance(self.permutation, list):
            self.permutation = list(self.permutation)
    
    def copy(self) -> 'MOSolution':
        """Create a deep copy of the solution"""
        new_solution = MOSolution(
            permutation=self.permutation.copy(),
            objectives=self.objectives,
            routes=[route.copy() for route in self.routes] if self.routes else None,
            feasible=self.feasible,
            constraint_violations=self.constraint_violations,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            generation=self.generation,
            evaluation_time=self.evaluation_time
        )
        # Reset NSGA-II specific lists for the copy
        new_solution.domination_count = 0
        new_solution.dominated_solutions = []
        return new_solution
    
    @property
    def total_distance(self) -> float:
        """Get total distance objective"""
        return self.objectives[0] if self.objectives else float('inf')
    
    @property
    def route_balance(self) -> float:
        """Get route balance (std deviation) objective"""
        return self.objectives[1] if self.objectives else float('inf')
    
    @property
    def n_customers(self) -> int:
        """Number of customers in the solution"""
        return len(self.permutation)
    
    @property
    def n_routes(self) -> int:
        """Number of routes in the decoded solution"""
        return len(self.routes) if self.routes else 0
    
    def distance_to(self, other: 'MOSolution') -> float:
        """Euclidean distance in objective space"""
        if not (self.objectives and other.objectives):
            return float('inf')
        
        return np.sqrt(
            (self.objectives[0] - other.objectives[0])**2 + 
            (self.objectives[1] - other.objectives[1])**2
        )
    
    def __str__(self) -> str:
        """String representation"""
        obj_str = f"({self.total_distance:.2f}, {self.route_balance:.2f})" if self.objectives else "No objectives"
        return f"MOSolution(obj={obj_str}, rank={self.rank}, feasible={self.feasible})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        """Equality comparison based on permutation"""
        if not isinstance(other, MOSolution):
            return False
        return self.permutation == other.permutation
    
    def __hash__(self) -> int:
        """Hash based on permutation"""
        return hash(tuple(self.permutation))


def create_random_solution(n_customers: int, seed: Optional[int] = None) -> MOSolution:
    """
    Create a random MOSolution with shuffled permutation
    
    Args:
        n_customers: Number of customers
        seed: Random seed for reproducibility
    
    Returns:
        Random MOSolution
    """
    if seed is not None:
        np.random.seed(seed)
    
    permutation = list(range(1, n_customers + 1))
    np.random.shuffle(permutation)
    
    return MOSolution(permutation=permutation)


def create_solution_from_permutation(permutation: List[int]) -> MOSolution:
    """
    Create MOSolution from existing permutation
    
    Args:
        permutation: Customer permutation
    
    Returns:
        MOSolution with given permutation
    """
    return MOSolution(permutation=permutation.copy())
