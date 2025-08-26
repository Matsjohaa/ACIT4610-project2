import random
from typing import List, Sequence


def ox_crossover(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> List[int]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))

    child: List[int] = [0] * n
    filled = [False] * n

    seg = list(p1[a:b + 1])
    child[a:b + 1] = seg
    for idx in range(a, b + 1):
        filled[idx] = True

    used = set(seg)
    j = (b + 1) % n

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
