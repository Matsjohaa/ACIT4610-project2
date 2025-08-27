import random
from typing import List, Sequence
from .constants import PARENTS_K, CROSSOVER_METHOD


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


def k_parent_crossover(parents: Sequence[Sequence[int]], rng: random.Random) -> List[int]:
    """
    Generalized crossover for k parents.
    Iteratively fold the list of parents using a 2-parent operator (OX by default):
      child = ox(p1, p2); child = ox(child, p3); ...; child = ox(child, pk)
    """
    if len(parents) == 0:
        raise ValueError("k_parent_crossover: need at least 1 parent")
    if len(parents) == 1:
        return list(parents[0])

    # Currently only OX is implemented; can be extended via CROSSOVER_METHOD
    if CROSSOVER_METHOD == "OX":
        op2 = ox_2parent
    # elif CROSSOVER_METHOD == "PMX":
    #     op2 = pmx_2parent
    else:
        raise ValueError(f"Unknown crossover method: {CROSSOVER_METHOD}")

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
