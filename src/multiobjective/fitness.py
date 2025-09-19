"""
Multi-objective fitness evaluation for Capacitated VRP
"""

import numpy as np
import time
from typing import Tuple

# Local imports
from .solution import MOSolution

# Parent module imports - handle both relative and absolute
try:
    from ..models import VRPInstance
    from ..split import dp_split_capacity
    from ..distances import route_length
except ImportError:
    # Fallback for direct execution - import from parent directory
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from models import VRPInstance
        from split import dp_split_capacity  
        from distances import route_length
    except ImportError as e:
        print(f"Warning: Could not import VRP modules: {e}")
        # Create dummy classes/functions for testing
        class VRPInstance:
            def __init__(self):
                self.customers = []
                self.n_vehicles = 1
                self.capacity = 100
        
        def dp_split_capacity(perm, n_vehicles, M, demands, capacity):
            # Dummy implementation - split into equal parts
            chunk_size = max(1, len(perm) // n_vehicles)
            return [perm[i:i+chunk_size] for i in range(0, len(perm), chunk_size)]
        
        def route_length(route, M):
            # Dummy implementation
            if not route:
                return 0.0
            total = M[0, route[0]] if M.shape[0] > route[0] else 10.0
            for i in range(len(route)-1):
                if M.shape[0] > max(route[i], route[i+1]):
                    total += M[route[i], route[i+1]]
                else:
                    total += 10.0
            total += M[route[-1], 0] if M.shape[0] > route[-1] else 10.0
            return total


def evaluate_mo_objectives(solution: MOSolution, instance: VRPInstance, distance_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate both objectives for Multi-Objective Capacitated VRP:
    1. Total Distance (minimize)
    2. Standard Deviation of Route Lengths (minimize) - Route Balance
    
    Args:
        solution: MOSolution to evaluate
        instance: VRP instance with customer demands and capacity
        distance_matrix: Distance matrix between all points (depot + customers)
    
    Returns:
        Tuple[float, float]: (total_distance, std_deviation)
    """
    start_time = time.time()
    
    try:
        # Extract customer demands (1-indexed customers -> 0-indexed demands)
        demands = [customer.demand for customer in instance.customers]
        
        # Split permutation into capacity-feasible routes
        routes = dp_split_capacity(
            solution.permutation,
            instance.n_vehicles,
            distance_matrix,
            demands,
            instance.capacity
        )
        
        # Store routes and mark as feasible
        solution.routes = routes
        solution.feasible = True
        solution.constraint_violations = 0
        
        # Calculate route lengths
        route_lengths = []
        for route in routes:
            if route:  # Skip empty routes
                length = route_length(route, distance_matrix)
                route_lengths.append(length)
        
        if not route_lengths:
            # No valid routes - should not happen with proper splitting
            solution.feasible = False
            solution.objectives = (1e6, 1e6)
            return solution.objectives
        
        # Objective 1: Total Distance (minimize)
        total_distance = sum(route_lengths)
        
        # Objective 2: Standard Deviation of Route Lengths (minimize)
        if len(route_lengths) > 1:
            mean_length = total_distance / len(route_lengths)
            variance = sum((length - mean_length)**2 for length in route_lengths) / len(route_lengths)
            std_deviation = np.sqrt(variance)
        else:
            # Only one route = perfect balance
            std_deviation = 0.0
        
        # Store objectives
        solution.objectives = (total_distance, std_deviation)
        
        # Record evaluation time
        solution.evaluation_time = time.time() - start_time
        
        return solution.objectives
        
    except (ValueError, Exception) as e:
        # Handle infeasible solutions or other errors
        solution.feasible = False
        solution.constraint_violations = 1
        solution.routes = []
        
        # Assign high penalty values for infeasible solutions
        penalty_distance = 1e6
        penalty_balance = 1e6
        solution.objectives = (penalty_distance, penalty_balance)
        
        solution.evaluation_time = time.time() - start_time
        
        return solution.objectives


def batch_evaluate(solutions: list[MOSolution], instance: VRPInstance, distance_matrix: np.ndarray) -> None:
    """
    Evaluate objectives for a batch of solutions
    
    Args:
        solutions: List of MOSolutions to evaluate
        instance: VRP instance
        distance_matrix: Distance matrix
    """
    for solution in solutions:
        evaluate_mo_objectives(solution, instance, distance_matrix)


def is_feasible_solution(solution: MOSolution, instance: VRPInstance) -> bool:
    """
    Check if a solution respects all VRP constraints
    
    Args:
        solution: Solution to check
        instance: VRP instance with constraints
    
    Returns:
        True if solution is feasible
    """
    if not solution.routes:
        return False
    
    demands = [customer.demand for customer in instance.customers]
    
    # Check capacity constraints for each route
    for route in solution.routes:
        if not route:  # Skip empty routes
            continue
            
        # Calculate total demand for this route
        route_demand = sum(demands[customer - 1] for customer in route)  # customers are 1-indexed
        
        if route_demand > instance.capacity:
            return False
    
    # Check that all customers are visited exactly once
    all_customers = set()
    for route in solution.routes:
        for customer in route:
            if customer in all_customers:
                return False  # Customer visited multiple times
            all_customers.add(customer)
    
    expected_customers = set(range(1, len(instance.customers) + 1))
    if all_customers != expected_customers:
        return False  # Not all customers visited or invalid customer IDs
    
    return True


def calculate_route_statistics(solution: MOSolution, distance_matrix: np.ndarray) -> dict:
    """
    Calculate detailed statistics for a solution's routes
    
    Args:
        solution: Solution to analyze
        distance_matrix: Distance matrix
    
    Returns:
        Dictionary with route statistics
    """
    if not solution.routes:
        return {
            'n_routes': 0,
            'route_lengths': [],
            'total_distance': float('inf'),
            'mean_route_length': 0,
            'std_route_length': 0,
            'min_route_length': 0,
            'max_route_length': 0
        }
    
    # Calculate route lengths
    route_lengths = []
    for route in solution.routes:
        if route:
            length = route_length(route, distance_matrix)
            route_lengths.append(length)
    
    if not route_lengths:
        return {
            'n_routes': 0,
            'route_lengths': [],
            'total_distance': 0,
            'mean_route_length': 0,
            'std_route_length': 0,
            'min_route_length': 0,
            'max_route_length': 0
        }
    
    total_distance = sum(route_lengths)
    mean_length = total_distance / len(route_lengths)
    std_length = np.sqrt(sum((l - mean_length)**2 for l in route_lengths) / len(route_lengths)) if len(route_lengths) > 1 else 0.0
    
    return {
        'n_routes': len(route_lengths),
        'route_lengths': route_lengths,
        'total_distance': total_distance,
        'mean_route_length': mean_length,
        'std_route_length': std_length,
        'min_route_length': min(route_lengths),
        'max_route_length': max(route_lengths)
    }


def objective_space_distance(sol1: MOSolution, sol2: MOSolution) -> float:
    """
    Calculate Euclidean distance between two solutions in objective space
    
    Args:
        sol1: First solution
        sol2: Second solution
    
    Returns:
        Euclidean distance in objective space
    """
    if not (sol1.objectives and sol2.objectives):
        return float('inf')
    
    return np.sqrt(
        (sol1.objectives[0] - sol2.objectives[0])**2 + 
        (sol1.objectives[1] - sol2.objectives[1])**2
    )


def normalize_objectives(solutions: list[MOSolution]) -> None:
    """
    Normalize objectives to [0, 1] range based on min/max values in the population
    
    Note: This modifies the solutions in-place and should be used carefully
    
    Args:
        solutions: List of solutions to normalize
    """
    if not solutions or not all(sol.objectives for sol in solutions):
        return
    
    # Find min and max for each objective
    obj1_values = [sol.objectives[0] for sol in solutions]
    obj2_values = [sol.objectives[1] for sol in solutions]
    
    obj1_min, obj1_max = min(obj1_values), max(obj1_values)
    obj2_min, obj2_max = min(obj2_values), max(obj2_values)
    
    # Normalize (handle case where min == max)
    for solution in solutions:
        if solution.objectives:
            if obj1_max > obj1_min:
                norm_obj1 = (solution.objectives[0] - obj1_min) / (obj1_max - obj1_min)
            else:
                norm_obj1 = 0.0
            
            if obj2_max > obj2_min:
                norm_obj2 = (solution.objectives[1] - obj2_min) / (obj2_max - obj2_min)
            else:
                norm_obj2 = 0.0
            
            solution.objectives = (norm_obj1, norm_obj2)
