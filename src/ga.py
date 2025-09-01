import time
import random
from typing import List, Callable
import numpy as np
from .constants import PARENTS_K
from .crossover import crossover_dispatch
from .mutation import mutation_dispatch
from .fitness import fitness_maximizing


def tournament_select(pop, fit, k, rng: random.Random):
    """
    Tournament selection (size=2 by default in evolve).
    Randomly sample two individuals and select the fitter one.
    Repeated k times to produce a parent pool of size k.
    """
    idx = list(range(len(pop)))
    winners = []
    for _ in range(k):
        i, j = rng.sample(idx, 2)
        winners.append(pop[i] if fit[i] >= fit[j] else pop[j])
    return winners


def init_population(n_customers: int, pop_size: int, rng: random.Random) -> List[List[int]]:
    """
    Initialize a random population of permutations of customer indices.
    Each individual is a permutation [1..n_customers].
    """
    base = list(range(1, n_customers + 1))
    pop = []
    for _ in range(pop_size):
        p = base[:]
        rng.shuffle(p)
        pop.append(p)
    return pop


def evolve(
        M: np.ndarray,
        n_vehicles: int,
        pop_size: int,
        generations: int,
        pc: float,
        pm: float,
        rng: random.Random,
        select_fn: Callable = tournament_select,
        fitness_fn: Callable = fitness_maximizing,
        record_history: bool = True,
):
    """
    Run the Genetic Algorithm loop:
    - initialize population
    - evaluate fitness
    - repeat for 'generations':
        * select parents
        * apply crossover and mutation
        * form new population
        * update global best
    Returns:
        best_ind : the best individual (permutation of customers)
        best_fit : the best fitness value
        info     : dictionary with runtime, evaluations, history, best distance estimate
    """
    # number of customers = size of permutation
    n_customers = M.shape[0] - 1
    pop = init_population(n_customers, pop_size, rng)

    # initial evaluation
    t0 = time.perf_counter()
    fit = [fitness_fn(ind, n_vehicles, M) for ind in pop]
    evaluations = len(pop)

    # track best individual
    best_ind, best_fit = max(zip(pop, fit), key=lambda t: t[1])
    history = [best_fit] if record_history else None

    # main GA loop
    for _ in range(generations):
        # select parents
        parents = select_fn(pop, fit, pop_size, rng)

        # create new children
        children = []
        group = PARENTS_K
        for i in range(0, pop_size, group):
            grp = [parents[(i + t) % pop_size] for t in range(group)]
            if rng.random() < pc:
                c1 = crossover_dispatch(grp, rng)
            else:
                c1 = grp[0][:]
            if rng.random() < pm:
                mutation_dispatch(c1, rng)
            children.append(c1)

        # new generation replaces old population
        pop = children[:pop_size]
        fit = [fitness_fn(ind, n_vehicles, M) for ind in pop]
        evaluations += len(pop)

        # update the best solution found
        cand_ind, cand_fit = max(zip(pop, fit), key=lambda t: t[1])
        if cand_fit > best_fit:
            best_ind, best_fit = cand_ind[:], cand_fit
        if record_history:
            history.append(best_fit)

    # return results and stats
    runtime_s = time.perf_counter() - t0
    best_distance = 1.0 / best_fit - 1.0
    info = {
        "runtime_s": runtime_s,
        "evaluations": evaluations,
        "fitness_history": history or [],
        "best_distance_est": best_distance,
    }
    return best_ind, best_fit, info
