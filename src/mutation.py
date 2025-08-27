import random
from typing import List
from .constants import MUTATION_METHOD


def mutate_swap(perm: List[int], rng: random.Random) -> List[int]:
    """
    Swap mutation:
    randomly choose two positions and swap their values.
    """
    if len(perm) < 2:
        return perm
    i, j = rng.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def mutate_inversion(perm: List[int], rng: random.Random) -> List[int]:
    """
    Inversion mutation:
    randomly select a subsequence and reverse its order.
    """
    if len(perm) < 2:
        return perm
    i, j = sorted(rng.sample(range(len(perm)), 2))
    perm[i:j + 1] = reversed(perm[i:j + 1])
    return perm


def mutate_insert(perm: List[int], rng: random.Random) -> List[int]:
    """
    Insert mutation:
    remove an element from one position and insert it at another position.
    """
    if len(perm) < 2:
        return perm
    i, j = rng.sample(range(len(perm)), 2)
    x = perm.pop(i)
    perm.insert(j, x)
    return perm


def mutation_dispatch(perm: List[int], rng: random.Random) -> List[int]:
    """
    Dispatcher function: chooses which mutation to apply based on MUTATION_METHOD.
    """
    if MUTATION_METHOD == "swap":
        return mutate_swap(perm, rng)
    elif MUTATION_METHOD == "inversion":
        return mutate_inversion(perm, rng)
    elif MUTATION_METHOD == "insert":
        return mutate_insert(perm, rng)
    else:
        raise ValueError(f"Unknown mutation method: {MUTATION_METHOD}")
