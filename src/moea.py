import random
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from .constants import (
    POP_SIZE, GENERATIONS, PC, PM, PARENTS_K,
    CROSSOVER_METHOD, MUTATION_METHOD, MOEA_ALGORITHM,
    LOCAL_SEARCH_PROB, ADAPTIVE_PM, PM_FLOOR
)
from .instances import InstanceData, load_instance
from .crossover import crossover_dispatch
from .mutation import mutation_dispatch
from .split import dp_split_capacity
from .distances import distance_matrix, route_length

# =============================
# Data structures
# =============================

@dataclass
class Individual:
    perm: List[int]
    objectives: Tuple[float, float] | None = None  # (total_distance, route_length_std)
    rank: int | None = None
    crowding: float = 0.0

# =============================
# Helpers
# =============================

def build_distance_matrix(inst: InstanceData):
    depot = inst.coords[0]
    customers = [inst.coords[i] for i in range(1, len(inst.demands) + 1)]
    M = distance_matrix(depot, [type('C', (), {'x': c[0], 'y': c[1], 'demand': d}) for c, d in zip(customers, inst.demands)])
    return M


def capacity_split(perm: List[int], M: np.ndarray, inst: InstanceData):
    return dp_split_capacity(perm, inst.n_vehicles, M, inst.demands, inst.capacity)


def evaluate_individual(ind: Individual, M: np.ndarray, inst: InstanceData):
    # attempt split; infeasible -> large penalty objectives
    try:
        routes = capacity_split(ind.perm, M, inst)
    except ValueError:
        ind.objectives = (1e9, 1e9)
        return
    total_dist = 0.0
    route_dists = []
    for r in routes:
        if not r:
            continue
        d = route_length(r, M)
        total_dist += d
        route_dists.append(d)
    if not route_dists:
        ind.objectives = (1e9, 1e9)
        return
    if len(route_dists) <= 1:
        std_dev = 0.0
        cv = 0.0
    else:
        mean = sum(route_dists)/len(route_dists)
        var = sum((x-mean)**2 for x in route_dists)/len(route_dists)
        std_dev = math.sqrt(var)
        cv = std_dev / mean if mean > 0 else 0.0
    ind.objectives = (total_dist, std_dev)

# =============================
# Local search (2-opt per route)
# =============================

def two_opt_route(route: List[int], M: np.ndarray):
    improved = True
    best = route[:]
    best_len = route_length(best, M)
    n = len(route)
    if n < 4:
        return route
    while improved:
        improved = False
        for i in range(0, n - 2):
            for k in range(i + 2, n):
                if k - i < 2:
                    continue
                new = best[:i+1] + list(reversed(best[i+1:k])) + best[k:]
                new_len = route_length(new, M)
                if new_len + 1e-9 < best_len:
                    best = new
                    best_len = new_len
                    improved = True
        # single pass improvement only to limit time
        break
    return best

def local_search(ind: Individual, M: np.ndarray, inst: InstanceData):
    # apply 2-opt inside each route then flatten back to permutation
    try:
        routes = capacity_split(ind.perm, M, inst)
    except ValueError:
        return
    new_perm: List[int] = []
    for r in routes:
        if not r:
            continue
        opt = two_opt_route(r, M)
        new_perm.extend(opt)
    # only accept if permutation length matches and differs
    if len(new_perm) == len(ind.perm) and new_perm != ind.perm:
        ind.perm = new_perm
        evaluate_individual(ind, M, inst)

# =============================
# Initialization & variation
# =============================

def init_population(rng: random.Random, inst: InstanceData) -> List[Individual]:
    base = list(range(1, len(inst.demands) + 1))
    pop: List[Individual] = []
    for _ in range(POP_SIZE):
        p = base[:]
        rng.shuffle(p)
        pop.append(Individual(perm=p))
    return pop


def tournament(pop: List[Individual], rng: random.Random) -> Individual:
    a, b = rng.sample(pop, 2)
    # lower objectives are better (minimization) -> use rank then crowding
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    # tie: prefer higher crowding
    return a if a.crowding > b.crowding else b


def make_child(parents: List[Individual], rng: random.Random, current_gen: int) -> Individual:
    p_perms = [p.perm for p in parents]
    child_perm = crossover_dispatch(p_perms, rng) if rng.random() < PC else p_perms[0][:]
    # adaptive mutation probability
    if ADAPTIVE_PM:
        t = current_gen / max(1, GENERATIONS - 1)
        pm_eff = (PM - PM_FLOOR) * (1 - t) + PM_FLOOR
    else:
        pm_eff = PM
    if rng.random() < pm_eff:
        mutation_dispatch(child_perm, rng)
    child = Individual(perm=child_perm)
    return child

# =============================
# NSGA-II Core
# =============================

def dominates(a: Individual, b: Individual) -> bool:
    assert a.objectives and b.objectives
    return all(x <= y for x, y in zip(a.objectives, b.objectives)) and any(x < y for x, y in zip(a.objectives, b.objectives))


def fast_non_dominated_sort(pop: List[Individual]):
    fronts: List[List[Individual]] = []
    S = {id(ind): [] for ind in pop}
    n = {id(ind): 0 for ind in pop}
    first: List[Individual] = []
    for p in pop:
        Sp = []
        np_ = 0
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                Sp.append(q)
            elif dominates(q, p):
                np_ += 1
        S[id(p)] = Sp
        n[id(p)] = np_
        if np_ == 0:
            p.rank = 0
            first.append(p)
    fronts.append(first)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: List[Individual] = []
        for p in fronts[i]:
            for q in S[id(p)]:
                n[id(q)] -= 1
                if n[id(q)] == 0:
                    q.rank = i + 1
                    nxt.append(q)
        i += 1
        if nxt:
            fronts.append(nxt)
    return fronts


def crowding_distance(front: List[Individual]):
    if not front:
        return
    m = len(front[0].objectives)
    for ind in front:
        ind.crowding = 0.0
    for obj_index in range(m):
        front.sort(key=lambda ind: ind.objectives[obj_index])
        front[0].crowding = front[-1].crowding = float('inf')
        vals = [ind.objectives[obj_index] for ind in front]
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            continue
        for i in range(1, len(front) - 1):
            prev_v = front[i - 1].objectives[obj_index]
            next_v = front[i + 1].objectives[obj_index]
            front[i].crowding += (next_v - prev_v) / (max_v - min_v)


def nsga2_step(pop: List[Individual], rng: random.Random, M: np.ndarray, inst: InstanceData, current_gen: int) -> List[Individual]:
    # Variation (binary tournament selection)
    offspring: List[Individual] = []
    while len(offspring) < POP_SIZE:
        parents = [tournament(pop, rng) for _ in range(PARENTS_K)]
        child = make_child(parents, rng, current_gen)
        evaluate_individual(child, M, inst)
        if rng.random() < LOCAL_SEARCH_PROB:
            local_search(child, M, inst)
        offspring.append(child)
    # Combine and sort
    combined = pop + offspring
    fronts = fast_non_dominated_sort(combined)
    new_pop: List[Individual] = []
    for f in fronts:
        crowding_distance(f)
        if len(new_pop) + len(f) <= POP_SIZE:
            new_pop.extend(f)
        else:
            f.sort(key=lambda ind: (-ind.crowding))
            new_pop.extend(f[:POP_SIZE - len(new_pop)])
            break
    return new_pop

# =============================
# VEGA (Vector Evaluated Genetic Algorithm)
# =============================

def vega_step(pop: List[Individual], rng: random.Random, M: np.ndarray, inst: InstanceData, current_gen: int) -> List[Individual]:
    # Split pop into sub-populations focusing on one objective each
    k = 2  # number of objectives
    sub_size = POP_SIZE // k
    parents_groups = []
    for obj_index in range(k):
        # select based on single objective obj_index
        sorted_pop = sorted(pop, key=lambda ind: ind.objectives[obj_index])
        parents_groups.append(sorted_pop[:sub_size])
    # Produce children uniformly
    children: List[Individual] = []
    while len(children) < POP_SIZE:
        group = rng.choice(parents_groups)
        parents = rng.sample(group, min(PARENTS_K, len(group)))
        child = make_child(parents, rng, current_gen)
        evaluate_individual(child, M, inst)
        if rng.random() < LOCAL_SEARCH_PROB:
            local_search(child, M, inst)
        children.append(child)
    # Evaluate crowding/rank for record (not used for selection here)
    combined = pop + children
    fronts = fast_non_dominated_sort(combined)
    for f in fronts:
        crowding_distance(f)
    return children

# =============================
# Main run function
# =============================

def run_moea(seed: int = 0, instance: InstanceData | None = None):
    rng = random.Random(seed)
    inst = instance if instance is not None else load_instance('small_01')
    M = build_distance_matrix(inst)
    pop = init_population(rng, inst)
    for ind in pop:
        evaluate_individual(ind, M, inst)
    # initial ranking for NSGA-II needs rank/crowding
    fronts = fast_non_dominated_sort(pop)
    for f in fronts:
        crowding_distance(f)

    for gen in range(GENERATIONS):
        if MOEA_ALGORITHM.upper() == 'NSGA2':
            pop = nsga2_step(pop, rng, M, inst, gen)
        elif MOEA_ALGORITHM.upper() == 'VEGA':
            pop = vega_step(pop, rng, M, inst, gen)
        else:
            raise ValueError(f"Unknown MOEA_ALGORITHM {MOEA_ALGORITHM}")
    # Final non-dominated set
    fronts = fast_non_dominated_sort(pop)
    pareto = fronts[0]
    return pareto, M, inst

__all__ = [
    'Individual', 'run_moea'
]
