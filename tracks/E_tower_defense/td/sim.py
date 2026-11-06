"""Deterministic tower-defense wave simulator (Track E).

Model (kept deliberately small so every number can be checked by hand):
  * Enemies walk along a fixed path of grid cells at `speed` path-cells per tick.
  * Each tick: spawn -> slow auras -> towers fire -> survivors move.
  * A damage tower hits the in-range enemy that is furthest along the path.
  * A slow tower (slow < 1) multiplies the speed of every in-range enemy; the
    strongest (smallest) slow wins, slows do not stack.
  * An enemy that reaches the last path cell has "leaked".
Everything is deterministic: same inputs -> same result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TowerType:
    name: str
    cost: int
    range: float          # Euclidean, in grid cells
    dps: float            # damage per tick (0 for pure slow towers)
    slow: float = 1.0     # speed multiplier applied to enemies in range (1.0 = none)


@dataclass(frozen=True)
class Enemy:
    hp: float
    speed: float          # path cells per tick (> 0)
    spawn: int            # tick at which it appears at path[0]


@dataclass(frozen=True)
class GameMap:
    width: int
    height: int
    path: tuple           # ((x, y), ...) ordered from spawn to exit

    def buildable(self):
        on_path = set(self.path)
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if (x, y) not in on_path]


@dataclass
class Result:
    damage: float         # total HP removed from enemies (overkill is not counted)
    leaked: int           # enemies that reached the exit
    leaked_hp: float     # HP they still had
    killed: int
    ticks: int


DEFAULT_TYPES = {
    "gun":    TowerType("gun", cost=10, range=2.5, dps=5.0),
    "cannon": TowerType("cannon", cost=25, range=2.0, dps=16.0),
    "frost":  TowerType("frost", cost=15, range=2.5, dps=0.0, slow=0.5),
}


def validate_placement(gmap: GameMap, placement: dict, types: dict, budget: float):
    """Raise ValueError if the placement breaks a constraint. Returns total cost."""
    ok_cells = set(gmap.buildable())
    cost = 0
    for cell, tname in placement.items():
        if cell not in ok_cells:
            raise ValueError(f"cell {cell} is not buildable")
        if tname not in types:
            raise ValueError(f"unknown tower type {tname!r}")
        cost += types[tname].cost
    if cost > budget:
        raise ValueError(f"cost {cost} exceeds budget {budget}")
    return cost


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def simulate(gmap: GameMap, placement: dict, types: dict, wave: list) -> Result:
    """Run one wave. `placement` maps (x, y) -> tower type name."""
    for e in wave:
        if e.speed <= 0 or e.hp <= 0:
            raise ValueError("enemies need hp > 0 and speed > 0")
    path = gmap.path
    last = len(path) - 1
    towers = [(cell, types[n]) for cell, n in sorted(placement.items())]
    pending = sorted(range(len(wave)), key=lambda i: (wave[i].spawn, i))
    nxt = 0
    active = []           # [index, pos, hp]
    damage = leaked_hp = 0.0
    leaked = killed = 0
    t = 0
    while nxt < len(pending) or active:
        while nxt < len(pending) and wave[pending[nxt]].spawn <= t:
            i = pending[nxt]
            active.append([i, 0.0, wave[i].hp])
            nxt += 1
        # 1. slow auras (decided from the position at the start of the tick)
        factor = {}
        for en in active:
            cell = path[min(int(en[1]), last)]
            f = 1.0
            for tcell, tt in towers:
                if tt.slow < 1.0 and _dist(tcell, cell) <= tt.range:
                    f = min(f, tt.slow)
            factor[en[0]] = f
        # 2. towers fire (furthest-along in-range enemy that is still alive)
        for tcell, tt in towers:
            if tt.dps <= 0:
                continue
            best = None
            for en in active:
                if en[2] <= 0:
                    continue
                cell = path[min(int(en[1]), last)]
                if _dist(tcell, cell) <= tt.range and (best is None or en[1] > best[1]):
                    best = en
            if best is not None:
                d = min(tt.dps, best[2])
                best[2] -= d
                damage += d
        # 3. remove dead, move the rest
        still = []
        for en in active:
            if en[2] <= 0:
                killed += 1
                continue
            en[1] += wave[en[0]].speed * factor[en[0]]
            if en[1] >= last:
                leaked += 1
                leaked_hp += en[2]
            else:
                still.append(en)
        active = still
        t += 1
    return Result(damage=damage, leaked=leaked, leaked_hp=leaked_hp, killed=killed, ticks=t)
