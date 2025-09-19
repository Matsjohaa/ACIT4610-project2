"""
Multi-Objective Optimization module for Capacitated VRP

This module provides implementations of Multi-Objective Evolutionary Algorithms (MOEAs)
for solving the Multi-Objective Capacitated Vehicle Routing Problem (MOCVRP).

Algorithms implemented:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- VEGA (Vector Evaluated Genetic Algorithm)

Objectives:
1. Total Distance (minimize)
2. Route Balance - Standard Deviation of Route Lengths (minimize)
"""

# Core multi-objective functionality
from .solution import MOSolution, create_random_solution, create_solution_from_permutation
from .dominance import dominates, non_dominated_sort, crowding_distance_assignment
from .fitness import evaluate_mo_objectives, batch_evaluate, is_feasible_solution

# NSGA-II algorithm
from .nsga2_clean import NSGA2_VRP_Clean

__all__ = [
    'MOSolution',
    'create_random_solution',
    'create_solution_from_permutation',
    'evaluate_mo_objectives',
    'batch_evaluate',
    'is_feasible_solution',
    'dominates',
    'non_dominated_sort',
    'crowding_distance_assignment',
    'NSGA2_VRP_Clean'
]