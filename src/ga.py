import random
from typing import List, Callable, Tuple
import numpy as np
from .crossover import ox_crossover
from .mutation import mutate_inversion
from .fitness import fitness_maximizing


# 2-way tournament selection, returns k parents
def tournament_select(pop, fit, k, rng: random.Random):
    idx = list(range(len(pop)))
    winners = []
    for _ in range(k):
        i, j = rng.sample(idx, 2)
        winners.append(pop[i] if fit[i] >= fit[j] else pop[j])
    return winners


def init_population(n_customers: int, pop_size: int, rng: random.Random) -> List[List[int]]:
    base = list(range(1, n_customers + 1))
    pop = []
    for _ in range(pop_size):
        p = base[:]
        rng.shuffle(p)
        pop.append(p)
    return pop


def evolve(M: np.ndarray, n_vehicles: int, pop_size: int, generations: int,
           pc: float, pm: float, rng: random.Random,
           select_fn: Callable = tournament_select,
           fitness_fn: Callable = fitness_maximizing):
    n_customers = M.shape[0] - 1
    pop = init_population(n_customers, pop_size, rng)
    fit = [fitness_fn(ind, n_vehicles, M) for ind in pop]

    # best-so-far (elitism record)
    best_ind, best_fit = max(zip(pop, fit), key=lambda t: t[1])

    for g in range(generations):
        # selection
        parents = select_fn(pop, fit, pop_size, rng)

        # variation
        children = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % pop_size]
            if rng.random() < pc:
                c1 = ox_crossover(p1, p2, rng)
                c2 = ox_crossover(p2, p1, rng)
            else:
                c1, c2 = p1[:], p2[:]
            # mutation
            if rng.random() < pm: mutate_inversion(c1, rng)
            if rng.random() < pm: mutate_inversion(c2, rng)
            children.extend([c1, c2])

        # evaluate
        pop = children[:pop_size]
        fit = [fitness_fn(ind, n_vehicles, M) for ind in pop]

        # elitism record
        cand_ind, cand_fit = max(zip(pop, fit), key=lambda t: t[1])
        if cand_fit > best_fit:
            best_ind, best_fit = cand_ind[:], cand_fit

    return best_ind, best_fit
