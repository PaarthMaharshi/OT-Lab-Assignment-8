"""Maps and enemy waves (seeded, reproducible)."""
from __future__ import annotations

import random

from .sim import Enemy, GameMap


def straight_map(length=10, height=3):
    """Horizontal path along y = 0. Used by the hand-computed tests."""
    return GameMap(length, height, tuple((x, 0) for x in range(length)))


def snake_map(width=15, height=11):
    """Serpentine path: rows y = 0, 2, 4, ... joined by one connector cell."""
    if width < 3 or height < 3:
        raise ValueError("snake map needs width >= 3 and height >= 3")
    path = []
    for k, y in enumerate(range(0, height, 2)):
        xs = range(width) if k % 2 == 0 else range(width - 1, -1, -1)
        path += [(x, y) for x in xs]
        if y + 2 < height:
            path.append((width - 1 if k % 2 == 0 else 0, y + 1))
    return GameMap(width, height, tuple(path))


def make_wave(seed=0, n=30, gap=2):
    """Mix of fast/weak scouts, normal grunts and slow/tanky brutes."""
    rng = random.Random(seed)
    kinds = [(20.0, 1.0), (60.0, 0.5), (200.0, 0.25)]   # (hp, speed)
    wave, t = [], 0
    for _ in range(n):
        hp, sp = rng.choices(kinds, weights=[3, 4, 1])[0]
        wave.append(Enemy(hp=hp, speed=sp, spawn=t))
        t += rng.randint(1, gap)
    return wave


def spiral_map(width=13, height=11):
    """Rectangular inward spiral with a one-cell gap between turns (a second geometry for the comparison)."""
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    path, seen, d, turns = [(0, 0)], {(0, 0)}, 0, 0
    while turns < 3:
        cx, cy = path[-1]
        nx, ny = cx + dirs[d][0], cy + dirs[d][1]
        nbrs = [(nx + a, ny + b) for a, b in dirs]
        ok = (0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen
              and all(n == (cx, cy) or n not in seen for n in nbrs))
        if ok:
            path.append((nx, ny)); seen.add((nx, ny)); turns = 0
        else:
            d = (d + 1) % 4; turns += 1
    return GameMap(width, height, tuple(path))
