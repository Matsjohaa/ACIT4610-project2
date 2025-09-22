import random
from typing import List, Sequence
from .constants import PARENTS_K, CROSSOVER_METHOD, MIXED_CROSSOVERS



def ox_2parent(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> List[int]:
    """Classic Order Crossover (OX) for TWO parents."""
    n = len(p1)
    # choose a random slice [a:b]
    a, b = sorted(rng.sample(range(n), 2))
    child: List[int] = [0] * n
    filled = [False] * n

    # copy segment from parent 1
    seg = list(p1[a:b + 1])
    child[a:b + 1] = seg
    for idx in range(a, b + 1):
        filled[idx] = True
    used = set(seg)
    j = (b + 1) % n

    # fill the rest with the positions using parent 2
    for x in p2:
        if x in used:
            continue
        while filled[j]:
            j = (j + 1) % n
        child[j] = x
        filled[j] = True
        used.add(x)
        j = (j + 1) % n
    assert all(filled), "OX produced an incompletely filled child"
    return child

def pmx_2parent(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> List[int]:
    """Partially Mapped Crossover (PMX) for TWO parents."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    # Copy slice from p1
    child[a:b+1] = p1[a:b+1]
    # Fill the rest from p2, respecting mapping
    for i in range(n):
        if child[i] is not None:
            continue
        val = p2[i]
        while val in child[a:b+1]:
            idx = p1.index(val)
            val = p2[idx]
        child[i] = val
    return child

def erx_2parent(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> List[int]:
    """Edge Recombination Crossover (ERX) for TWO parents."""
    n = len(p1)
    # Build edge map
    edge_map = {v: set() for v in p1}
    for parent in [p1, p2]:
        for i in range(n):
            left = parent[i-1]
            right = parent[(i+1)%n]
            edge_map[parent[i]].update([left, right])
    child = []
    current = rng.choice(p1)
    while len(child) < n:
        child.append(current)
        for edges in edge_map.values():
            edges.discard(current)
        if len(child) == n:
            break
        if edge_map[current]:
            next_candidates = list(edge_map[current])
            min_edges = min(len(edge_map[v]) for v in next_candidates)
            candidates = [v for v in next_candidates if len(edge_map[v]) == min_edges]
            current = rng.choice(candidates)
        else:
            unused = [v for v in p1 if v not in child]
            if unused:
                current = rng.choice(unused)
            else:
                break
    return child



def k_parent_crossover(parents: Sequence[Sequence[int]], rng: random.Random) -> List[int]:
    """
    Generalized crossover for k parents.
    Iteratively fold the list of parents using a 2-parent operator (method chosen by CROSSOVER_METHOD):
      child = op2(p1, p2); child = op2(child, p3); ...; child = op2(child, pk)
    """
    if len(parents) == 0:
        raise ValueError("k_parent_crossover: need at least 1 parent")
    if len(parents) == 1:
        return list(parents[0])

    method = CROSSOVER_METHOD
    if method == "mixed":
        method = rng.choice(MIXED_CROSSOVERS)

    if method == "OX":
        op2 = ox_2parent
    elif method == "PMX":
        op2 = pmx_2parent
    elif method == "ERX":
        op2 = erx_2parent
    else:
        raise ValueError(f"Unknown crossover method: {method}")

    child = list(parents[0])
    for p in parents[1:]:
        child = op2(child, p, rng)
    return child


def crossover_dispatch(parents: Sequence[Sequence[int]], rng: random.Random) -> List[int]:
    """
    Dispatcher: reads PARENTS_K from constants and applies k-parent crossover.
    If the number of provided parents differs from PARENTS_K,
    it still folds whatever was passed in.
    """
    chosen = parents[:PARENTS_K]
    return k_parent_crossover(chosen, rng)
