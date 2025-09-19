"""
Clean NSGA-II implementation following the exact algorithm steps
Implements the standard NSGA-II for Multi-Objective VRP
"""

import random
import time
import numpy as np
from typing import List, Tuple, Dict, Any

from .solution import MOSolution
from .fitness import evaluate_mo_objectives
from .dominance import non_dominated_sort, crowding_distance_assignment

# Parent module imports with fallback for testing
try:
    from ..models import VRPInstance
    from ..crossover import ox_2parent
    from ..mutation import mutate_inversion
except ImportError:
    print("Warning: Could not import VRP modules - using fallback implementations")
    
    class VRPInstance:
        def __init__(self):
            self.customers = []
            self.n_vehicles = 1
            self.capacity = 100
    
    def ox_2parent(p1, p2, rng):
        """Fallback Order Crossover"""
        n = len(p1)
        if n < 2:
            return p1[:]
        a, b = sorted(rng.sample(range(n), 2))
        child = [0] * n
        child[a:b+1] = p1[a:b+1]
        remaining = [x for x in p2 if x not in child[a:b+1]]
        j = 0
        for i in range(n):
            if child[i] == 0:
                child[i] = remaining[j]
                j += 1
        return child
    
    def mutate_inversion(perm, rng):
        """Fallback Inversion Mutation"""
        if len(perm) < 2:
            return perm[:]
        a, b = sorted(rng.sample(range(len(perm)), 2))
        result = perm[:]
        result[a:b+1] = reversed(result[a:b+1])
        return result


class NSGA2_VRP_Clean:
    """Clean NSGA-II implementation following exact algorithm steps"""
    
    def __init__(self, instance: VRPInstance, distance_matrix: np.ndarray, 
                 pop_size: int = 50, generations: int = 100,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.1,
                 niching_radius: float = 0.1, max_pareto_size: int = 10,
                 seed: int = None, verbose: bool = True):
        
        self.instance = instance
        self.distance_matrix = distance_matrix
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        
        # Niching parameters
        self.niching_radius = niching_radius
        self.max_pareto_size = max_pareto_size
        
        # Initialize RNG
        self.rng = random.Random(seed)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Statistics tracking
        self.generation_stats = []

    def step1_initialization(self) -> List[MOSolution]:
        """
        Step 1: Generate initial population of size N (random solutions)
        """
        if self.verbose:
            print("Step 1: Initializing population...")
        
        base = list(range(1, len(self.instance.customers) + 1))
        population = []
        
        for i in range(self.pop_size):
            permutation = base[:]
            self.rng.shuffle(permutation)
            solution = MOSolution(permutation=permutation, generation=0)
            population.append(solution)
        
        # Evaluate all objectives
        self.evaluate_objectives(population)
        
        if self.verbose:
            print(f"   Created {len(population)} random solutions")
        
        return population

    def evaluate_objectives(self, population: List[MOSolution]) -> None:
        """Evaluate objectives for unevaluated solutions"""
        for solution in population:
            if solution.objectives is None:
                evaluate_mo_objectives(solution, self.instance, self.distance_matrix)

    def step2_non_dominated_sorting(self, population: List[MOSolution]) -> List[List[MOSolution]]:
        """
        Step 2: Sort population into fronts based on dominance
        """
        fronts = non_dominated_sort(population)
        
        if self.verbose:
            front_sizes = [len(front) for front in fronts]
            print(f"Step 2: Non-dominated sorting -> {len(fronts)} fronts: {front_sizes}")
        
        return fronts

    def step3_crowding_distance(self, fronts: List[List[MOSolution]]) -> None:
        """
        Step 3: Calculate crowding distance for each front
        """
        if self.verbose:
            print("Step 3: Calculating crowding distances...")
        
        for front in fronts:
            if len(front) > 2:
                crowding_distance_assignment(front)

    def calculate_objective_distance(self, sol1: MOSolution, sol2: MOSolution) -> float:
        """Calculate normalized distance in objective space for niching"""
        if not (sol1.objectives and sol2.objectives):
            return float('inf')
        
        # Simple Euclidean distance in normalized objective space
        obj1 = np.array(sol1.objectives)
        obj2 = np.array(sol2.objectives)
        
        # Normalize by rough ranges (can be improved with population statistics)
        obj1_norm = obj1 / np.array([1000.0, 50.0])  # Rough VRP ranges
        obj2_norm = obj2 / np.array([1000.0, 50.0])
        
        return np.linalg.norm(obj1_norm - obj2_norm)

    def apply_niching(self, solutions: List[MOSolution]) -> List[MOSolution]:
        """Apply niching to reduce similar solutions in Pareto front"""
        if len(solutions) <= self.max_pareto_size:
            return solutions
        
        if self.verbose:
            print(f"   Applying niching: {len(solutions)} → {self.max_pareto_size} solutions")
        
        selected = []
        remaining = solutions[:]
        
        # 1. Add extreme solutions first (boundary points)
        if remaining:
            # Best distance solution
            best_dist = min(remaining, key=lambda x: x.objectives[0])
            selected.append(best_dist)
            remaining.remove(best_dist)
        
        if remaining:
            # Best balance solution  
            best_balance = min(remaining, key=lambda x: x.objectives[1])
            if best_balance not in selected:  # Avoid duplicates
                selected.append(best_balance)
                remaining.remove(best_balance)
        
        # 2. Add diverse solutions using niching
        while len(selected) < self.max_pareto_size and remaining:
            best_candidate = None
            max_min_distance = -1
            
            for candidate in remaining:
                # Calculate minimum distance to already selected solutions
                min_distance = min(
                    self.calculate_objective_distance(candidate, selected_sol)
                    for selected_sol in selected
                )
                
                # Select the candidate with maximum minimum distance (most diverse)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate and max_min_distance > self.niching_radius:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # If no solution is far enough, pick the best by crowding distance
                if remaining:
                    best_crowding = max(remaining, key=lambda x: x.crowding_distance)
                    selected.append(best_crowding)
                    remaining.remove(best_crowding)
        
        return selected

    def step4_selection(self, population: List[MOSolution]) -> MOSolution:
        """
        Step 4: Enhanced tournament selection with better selection pressure
        """
        # Use 3-way tournament for better selection pressure
        tournament_size = min(3, len(population))
        candidates = self.rng.sample(population, tournament_size)
        
        # Ensure ranks are set
        for candidate in candidates:
            if candidate.rank is None:
                candidate.rank = float('inf')
        
        # Sort by rank first, then by crowding distance, then by sum of objectives
        candidates.sort(key=lambda x: (
            x.rank,
            -x.crowding_distance,
            sum(x.objectives) if x.objectives else float('inf'),  # Tie-breaker
            self.rng.random()  # Final tie-breaker
        ))
        
        return candidates[0].copy()

    def step5_variation(self, population: List[MOSolution]) -> List[MOSolution]:
        """
        Step 5: Crossover + Mutation to generate offspring population of size N
        """
        if self.verbose:
            print("Step 5: Creating offspring through crossover and mutation...")
        
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Select parents
            parent1 = self.step4_selection(population)
            parent2 = self.step4_selection(population)
            
            # Crossover
            if self.rng.random() < self.crossover_rate:
                child1_perm = ox_2parent(parent1.permutation, parent2.permutation, self.rng)
                child2_perm = ox_2parent(parent2.permutation, parent1.permutation, self.rng)
            else:
                child1_perm = parent1.permutation[:]
                child2_perm = parent2.permutation[:]
            
            # Create children
            child1 = MOSolution(permutation=child1_perm)
            child2 = MOSolution(permutation=child2_perm)
            
            # Mutation
            if self.rng.random() < self.mutation_rate:
                child1.permutation = mutate_inversion(child1.permutation, self.rng)
            if self.rng.random() < self.mutation_rate:
                child2.permutation = mutate_inversion(child2.permutation, self.rng)
            
            offspring.extend([child1, child2])
        
        # Evaluate objectives for offspring
        self.evaluate_objectives(offspring[:self.pop_size])
        
        return offspring[:self.pop_size]

    def step7_environmental_selection(self, combined_population: List[MOSolution]) -> List[MOSolution]:
        """
        Step 7: Environmental selection - fill new population from best fronts
        """
        if self.verbose:
            print(f"Step 7: Environmental selection from {len(combined_population)} solutions...")
        
        # Sort combined population into fronts
        fronts = self.step2_non_dominated_sorting(combined_population)
        self.step3_crowding_distance(fronts)
        
        new_population = []
        
        # Fill population front by front
        # Fill population front by front - NO NICHING during evolution
        for front_idx, front in enumerate(fronts):
            # REMOVED: No niching during evolution - keep full Pareto fronts for diversity
            # if front_idx == 0 and len(front) > self.max_pareto_size:
            #     front = self.apply_niching(front)
            
            if len(new_population) + len(front) <= self.pop_size:
                # Take entire front
                new_population.extend(front)
                if self.verbose:
                    print(f"   Added entire front {front_idx+1}: {len(front)} solutions")
            else:
                # Partial front - select by crowding distance
                remaining = self.pop_size - len(new_population)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                selected = front[:remaining]
                new_population.extend(selected)
                if self.verbose:
                    print(f"   Added {remaining}/{len(front)} from front {front_idx+1} (by crowding distance)")
                break
        
        if self.verbose:
            print(f"   New population size: {len(new_population)}")
        
        return new_population

    def collect_generation_stats(self, population: List[MOSolution], generation: int) -> Dict[str, Any]:
        """Collect statistics for current generation"""
        feasible = [sol for sol in population if sol.feasible]
        
        if not feasible:
            return {
                'generation': generation,
                'feasible_count': 0,
                'pareto_size': 0,
                'best_distance': float('inf'),
                'best_balance': float('inf')
            }
        
        # Get actual Pareto front (rank 1 solutions)
        pareto_front = [sol for sol in feasible if sol.rank == 1]
        distances = [sol.objectives[0] for sol in feasible]
        balances = [sol.objectives[1] for sol in feasible]
        
        return {
            'generation': generation,
            'feasible_count': len(feasible),
            'pareto_size': len(pareto_front),
            'best_distance': min(distances),
            'best_balance': min(balances),
            'avg_distance': np.mean(distances),
            'avg_balance': np.mean(balances)
        }

    def run(self) -> Tuple[List[MOSolution], List[MOSolution], Dict[str, Any]]:
        """
        Main NSGA-II algorithm following exact steps
        
        ANTI-CONVERGENCE MODIFICATION:
        - NO niching during evolution (keeps full Pareto fronts for diversity)
        - Niching applied ONLY at the end for final reporting
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"🚀 Starting Clean NSGA-II with Niching: {self.pop_size} pop, {self.generations} gen")
            print(f"   Niching radius: {self.niching_radius}")
            print(f"   Max Pareto size: {self.max_pareto_size}")
            print("=" * 60)
        
        # Step 1: Initialize population P of size N
        population = self.step1_initialization()
        
        # Initial sorting and crowding distance
        fronts = self.step2_non_dominated_sorting(population)
        self.step3_crowding_distance(fronts)
        
        # Evolution loop
        for generation in range(self.generations):
            if self.verbose and generation % 25 == 0:
                print(f"\n--- Generation {generation} ---")
            
            # Step 5: Generate offspring Q through selection, crossover, mutation
            offspring = self.step5_variation(population)
            
            # Step 6: Combine parents + offspring (R = P ∪ Q, size 2N)
            combined_population = population + offspring
            
            # Step 7: Environmental selection - create new population P
            population = self.step7_environmental_selection(combined_population)
            
            # Collect statistics
            stats = self.collect_generation_stats(population, generation)
            self.generation_stats.append(stats)
            
            if self.verbose and generation % 25 == 0:
                print(f"   Pareto front: {stats['pareto_size']} solutions")
                print(f"   Best distance: {stats['best_distance']:.1f}")
                print(f"   Best balance: {stats['best_balance']:.2f}")
        
        # Final sorting to get Pareto front
        final_fronts = self.step2_non_dominated_sorting(population)
        self.step3_crowding_distance(final_fronts)
        
        # Get actual Pareto front (first front only)
        pareto_front = final_fronts[0] if final_fronts else []
        pareto_front = [sol for sol in pareto_front if sol.feasible]
        
        # Apply niching ONLY at the end for reporting (not during evolution)
        if len(pareto_front) > self.max_pareto_size:
            if self.verbose:
                print(f"\n🎯 Final niching: {len(pareto_front)} → {self.max_pareto_size} solutions")
            pareto_front = self.apply_niching(pareto_front)
        
        # Compile results
        runtime = time.time() - start_time
        run_info = {
            'algorithm': 'NSGA-II-Clean',
            'runtime_seconds': runtime,
            'total_evaluations': self.generations * self.pop_size * 2,
            'final_pareto_size': len(pareto_front),
            'final_feasible_count': len([sol for sol in population if sol.feasible]),
            'generation_stats': self.generation_stats,
            'parameters': {
                'pop_size': self.pop_size,
                'generations': self.generations,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'niching_radius': self.niching_radius,
                'max_pareto_size': self.max_pareto_size
            }
        }
        
        if self.verbose:
            print(f"\n✅ Clean NSGA-II completed in {runtime:.2f}s")
            print(f"   Final Pareto front: {len(pareto_front)} solutions")
            print(f"   Total fronts: {len(final_fronts)}")
        
        return population, pareto_front, run_info


if __name__ == "__main__":
    # Quick test
    print("🧪 Quick test of Clean NSGA-II...")
    
    class DummyCustomer:
        def __init__(self, demand):
            self.demand = demand
    
    instance = VRPInstance()
    instance.customers = [DummyCustomer(5) for _ in range(6)]
    instance.n_vehicles = 2
    instance.capacity = 20
    
    distance_matrix = np.random.rand(7, 7) * 100
    for i in range(7):
        distance_matrix[i, i] = 0
        for j in range(i+1, 7):
            distance_matrix[j, i] = distance_matrix[i, j]
    
    nsga2 = NSGA2_VRP_Clean(
        instance=instance,
        distance_matrix=distance_matrix,
        pop_size=60,          # Larger population
        generations=200,      # More generations  
        crossover_rate=0.8,   # Slightly lower crossover
        mutation_rate=0.25,   # Higher mutation
        niching_radius=0.2,   # Larger niching radius
        max_pareto_size=None,   # Allow more solutions
        seed=42,
        verbose=True
    )
    
    population, pareto_front, info = nsga2.run()
    print(f"✅ Test completed: {len(pareto_front)} Pareto solutions")
