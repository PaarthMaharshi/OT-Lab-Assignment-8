"""Baseline placements."""
from __future__ import annotations

import random


def random_placement(gmap, types, budget, rng: random.Random):
    """Random valid placement: keep adding a random tower on a random free cell
    while something is still affordable."""
    free = gmap.buildable()
    rng.shuffle(free)
    placement, spent = {}, 0
    names = list(types)
    while free:
        afford = [n for n in names if spent + types[n].cost <= budget]
        if not afford:
            break
        cell = free.pop()
        name = rng.choice(afford)
        placement[cell] = name
        spent += types[name].cost
    return placement
