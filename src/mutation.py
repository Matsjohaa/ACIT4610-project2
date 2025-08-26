import random
from typing import List


def mutate_swap(perm: List[int], rng: random.Random):
    if len(perm) < 2: return perm
    i, j = rng.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def mutate_inversion(perm: List[int], rng: random.Random):
    if len(perm) < 2: return perm
    i, j = sorted(rng.sample(range(len(perm)), 2))
    perm[i:j + 1] = reversed(perm[i:j + 1])
    return perm


def mutate_insert(perm: List[int], rng: random.Random):
    if len(perm) < 2: return perm
    i, j = rng.sample(range(len(perm)), 2)
    x = perm.pop(i)
    perm.insert(j, x)
    return perm
