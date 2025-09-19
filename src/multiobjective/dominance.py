"""
Pareto dominance operations for multi-objective optimization
"""

from typing import List
from .solution import MOSolution


def dominates(solution1: MOSolution, solution2: MOSolution) -> bool:
    """
    Check if solution1 dominates solution2 (both objectives are minimized)
    
    Solution A dominates solution B if:
    - A is better or equal in all objectives
    - A is strictly better in at least one objective
    
    Args:
        solution1: First solution
        solution2: Second solution
    
    Returns:
        True if solution1 dominates solution2
    """
    if not (solution1.objectives and solution2.objectives):
        # If either solution doesn't have objectives, handle based on feasibility
        if not solution1.feasible and solution2.feasible:
            return False
        if solution1.feasible and not solution2.feasible:
            return True
        return False
    
    # Handle infeasible solutions
    if not solution1.feasible and solution2.feasible:
        return False
    if solution1.feasible and not solution2.feasible:
        return True
    if not solution1.feasible and not solution2.feasible:
        return False  # Neither dominates if both are infeasible
    
    # Both solutions are feasible - check Pareto dominance
    obj1 = solution1.objectives
    obj2 = solution2.objectives
    
    # Check if solution1 is better or equal in all objectives
    better_or_equal = all(obj1[i] <= obj2[i] for i in range(len(obj1)))
    
    # Check if solution1 is strictly better in at least one objective
    strictly_better = any(obj1[i] < obj2[i] for i in range(len(obj1)))
    
    return better_or_equal and strictly_better


def non_dominated_sort(population: List[MOSolution]) -> List[List[MOSolution]]:
    """
    Perform non-dominated sorting on a population
    
    Returns a list of fronts, where each front is a list of non-dominated solutions.
    Front 0 contains the best solutions (rank 1), front 1 contains rank 2, etc.
    
    Args:
        population: List of solutions to sort
    
    Returns:
        List of fronts, each containing solutions of the same rank
    """
    # Reset domination attributes
    for solution in population:
        solution.domination_count = 0
        solution.dominated_solutions = []
        solution.rank = None
    
    # Calculate domination relationships
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i != j:
                if dominates(p, q):
                    p.dominated_solutions.append(q)
                elif dominates(q, p):
                    p.domination_count += 1
        
        # Solutions with domination_count = 0 belong to first front
        if p.domination_count == 0:
            p.rank = 1
    
    # Build fronts
    fronts = []
    current_rank = 1
    
    while True:
        current_front = [p for p in population if p.rank == current_rank]
        if not current_front:
            break
        
        fronts.append(current_front)
        
        # Process next front
        for p in current_front:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = current_rank + 1
        
        current_rank += 1
    
    return fronts


def is_pareto_front(solutions: List[MOSolution]) -> bool:
    """
    Check if a set of solutions forms a valid Pareto front
    (no solution dominates any other in the set)
    
    Args:
        solutions: List of solutions to check
    
    Returns:
        True if solutions form a Pareto front
    """
    for i, sol1 in enumerate(solutions):
        for j, sol2 in enumerate(solutions):
            if i != j and dominates(sol1, sol2):
                return False
    return True


def filter_non_dominated(solutions: List[MOSolution]) -> List[MOSolution]:
    """
    Filter a list of solutions to keep only non-dominated ones
    
    Args:
        solutions: List of solutions to filter
    
    Returns:
        List containing only non-dominated solutions
    """
    non_dominated = []
    
    for candidate in solutions:
        is_dominated = False
        
        # Check if candidate is dominated by any solution in non_dominated
        for existing in non_dominated:
            if dominates(existing, candidate):
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove any solutions in non_dominated that are dominated by candidate
            non_dominated = [sol for sol in non_dominated if not dominates(candidate, sol)]
            non_dominated.append(candidate)
    
    return non_dominated


def crowding_distance_assignment(solutions: List[MOSolution]) -> None:
    """
    Assign crowding distance to solutions (modifies solutions in-place)
    
    Crowding distance measures how close a solution is to its neighbors
    in the objective space. Used for diversity preservation.
    
    Args:
        solutions: List of solutions to assign crowding distances
    """
    n = len(solutions)
    
    # Initialize crowding distances
    for solution in solutions:
        solution.crowding_distance = 0.0
    
    if n <= 2:
        # Boundary solutions get infinite distance
        for solution in solutions:
            solution.crowding_distance = float('inf')
        return
    
    # For each objective
    n_objectives = 2  # We have exactly 2 objectives
    
    for obj_index in range(n_objectives):
        # Sort solutions by this objective
        solutions.sort(key=lambda x: x.objectives[obj_index] if x.objectives else float('inf'))
        
        # Boundary solutions get infinite distance
        solutions[0].crowding_distance = float('inf')
        solutions[-1].crowding_distance = float('inf')
        
        # Calculate objective range
        if solutions[-1].objectives and solutions[0].objectives:
            obj_range = solutions[-1].objectives[obj_index] - solutions[0].objectives[obj_index]
            
            if obj_range > 0:
                # Assign crowding distance to intermediate solutions
                for i in range(1, n - 1):
                    if (solutions[i].crowding_distance != float('inf') and 
                        solutions[i+1].objectives and solutions[i-1].objectives):
                        
                        distance = (solutions[i+1].objectives[obj_index] - 
                                  solutions[i-1].objectives[obj_index]) / obj_range
                        solutions[i].crowding_distance += distance


def crowded_comparison(solution1: MOSolution, solution2: MOSolution) -> int:
    """
    Compare two solutions based on rank and crowding distance
    
    Returns:
        -1 if solution1 is better
         1 if solution2 is better  
         0 if they are equivalent
    """
    if solution1.rank is None or solution2.rank is None:
        # Handle unranked solutions
        if solution1.rank is None and solution2.rank is not None:
            return 1
        if solution1.rank is not None and solution2.rank is None:
            return -1
        return 0
    
    # Compare by rank first (lower rank is better)
    if solution1.rank < solution2.rank:
        return -1
    elif solution1.rank > solution2.rank:
        return 1
    else:
        # Same rank - compare by crowding distance (higher is better)
        if solution1.crowding_distance > solution2.crowding_distance:
            return -1
        elif solution1.crowding_distance < solution2.crowding_distance:
            return 1
        else:
            return 0
