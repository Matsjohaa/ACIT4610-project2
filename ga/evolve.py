from .crossover import route_based_crossover
from .mutation import mutate
from .population import Population
from .constants import POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE

def evolve(dataset_name):
    population = Population(dataset_name, POPULATION_SIZE)

    for generation in range(NUM_GENERATIONS):
        new_population = []

        # Elitism: keep best solution
        elite = population.get_best_solution()
        new_population.append(elite)

        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover + Mutation
            child = route_based_crossover(parent1, parent2, elite.routes[0].depot)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)

        # Replace old with new
        population.solutions = new_population

        print(f"Gen {generation+1} Best Fitness: {elite.fitness():.4f}")

    return population.get_best_solution()

def tournament_selection(population, k=3):
    import random
    selected = random.sample(population.solutions, k)
    return max(selected, key=lambda s: s.fitness())
