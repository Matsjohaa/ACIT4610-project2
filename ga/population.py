from .solution import Solution, create_random_solution
from .constants import POPULATION_SIZE

class Population:
    def __init__(self, dataset_name: str, pop_size: int = POPULATION_SIZE):
        self.solutions = [create_random_solution(dataset_name) for _ in range(pop_size)]

    def get_best_solution(self):
        return max(self.solutions, key=lambda s: s.fitness())

pop = Population("small_1", 10)
for i, sol in enumerate(pop.solutions):
    print(f"Solution {i+1} Fitness: {sol.fitness():.4f}")

best = pop.get_best_solution()
print("\nBest solution:")
print(best)
print(f"Best fitness: {best.fitness():.4f}")