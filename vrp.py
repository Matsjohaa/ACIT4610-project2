from scenarios import get_scenarios
from ga import VRPGenetic
import random

# GA parameters
POP_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
TOURNAMENT_K = 3

if __name__ == "__main__":
    scenarios = get_scenarios()
    '''
        This will run the GA for all scenarios.
        To run the GA for only one scenario, replace for scenario in [scenario] with scenario = scenarios[0]  # or any index 0-5
    '''
    for scenario in scenarios:
        print(f"\nRunning GA for scenario: {scenario['name']}")
        vrp = VRPGenetic(scenario)
        population = vrp.initial_population(POP_SIZE)
        best_solution = None
        best_fitness = float('inf')
        for gen in range(GENERATIONS):
            new_population = []
            while len(new_population) < POP_SIZE:
                parent1 = vrp.selection(population, TOURNAMENT_K)
                parent2 = vrp.selection(population, TOURNAMENT_K)
                if random.random() < CROSSOVER_RATE:
                    child = vrp.crossover(parent1, parent2)
                else:
                    child = [route[:] for route in parent1]
                vrp.mutate(child, MUTATION_RATE)
                new_population.append(child)
            population = new_population
            gen_best = min(population, key=vrp.fitness)
            gen_best_fitness = vrp.fitness(gen_best)
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_solution = gen_best
            if (gen+1) % 10 == 0 or gen == 0:
                print(f"  Generation {gen+1}: Best fitness = {best_fitness:.2f}")
        print("Best solution found:")
        print(best_solution)
        print(f"Total distance: {best_fitness:.2f}")
