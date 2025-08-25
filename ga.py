import random
from typing import List, Dict, Any
import math

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

class VRPGenetic:
    def __init__(self, scenario: Dict[str, Any]):
        self.depot = scenario['depot']
        self.customers = scenario['customers']
        self.num_vehicles = scenario['num_vehicles']
        self.num_customers = len(self.customers)

    def create_chromosome(self) -> List[List[int]]:
        # Randomly assign customers to vehicles
        customer_indices = list(range(self.num_customers))
        random.shuffle(customer_indices)
        # Split customers as evenly as possible among vehicles
        routes = [[] for _ in range(self.num_vehicles)]
        for i, idx in enumerate(customer_indices):
            routes[i % self.num_vehicles].append(idx)
        return routes

    def initial_population(self, pop_size: int) -> List[List[List[int]]]:
        return [self.create_chromosome() for _ in range(pop_size)]

    def route_distance(self, route: List[int]) -> float:
        if not route:
            return 0.0
        dist = 0.0
        prev = self.depot
        for idx in route:
            dist += euclidean_distance(prev, self.customers[idx])
            prev = self.customers[idx]
        dist += euclidean_distance(prev, self.depot)  # return to depot
        return dist

    def fitness(self, chromosome: List[List[int]]) -> float:
        # Lower is better (total distance)
        return sum(self.route_distance(route) for route in chromosome)

    def selection(self, population: List[List[List[int]]], k: int = 3) -> List[List[int]]:
        # Select k random individuals and return the best (lowest distance)
        selected = random.sample(population, k)
        selected.sort(key=self.fitness)
        return selected[0]

    def crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        # Route-based crossover: randomly select routes from each parent
        num_routes = len(parent1)
        child = [[] for _ in range(num_routes)]
        assigned = set()
        for i in range(num_routes):
            if random.random() < 0.5:
                child[i] = parent1[i][:]
            else:
                child[i] = parent2[i][:]
            assigned.update(child[i])
        # Fix missing/duplicate customers
        all_customers = set(range(self.num_customers))
        missing = list(all_customers - assigned)
        # Remove duplicates
        seen = set()
        for route in child:
            i = 0
            while i < len(route):
                if route[i] in seen:
                    route.pop(i)
                else:
                    seen.add(route[i])
                    i += 1
        # Add missing customers to random routes
        for m in missing:
            idx = random.randint(0, num_routes - 1)
            child[idx].append(m)
        return child

    def mutate(self, chromosome: List[List[int]], mutation_rate: float = 0.1) -> None:
        # Swap two customers between or within routes
        if random.random() < mutation_rate:
            # Choose two routes
            non_empty_routes = [r for r in chromosome if r]
            if len(non_empty_routes) < 1:
                return
            route1 = random.choice(non_empty_routes)
            route2 = random.choice(non_empty_routes)
            idx1 = random.randint(0, len(route1) - 1)
            idx2 = random.randint(0, len(route2) - 1)
            # Swap
            route1[idx1], route2[idx2] = route2[idx2], route1[idx1]

# Example usage:
if __name__ == "__main__":
    from scenarios import get_scenarios
    scenarios = get_scenarios()
    vrp = VRPGenetic(scenarios[0])
    pop = vrp.initial_population(5)
    for i, chrom in enumerate(pop):
        print(f"Chromosome {i+1}: {chrom}")
        print(f"  Fitness (total distance): {vrp.fitness(chrom):.2f}")
