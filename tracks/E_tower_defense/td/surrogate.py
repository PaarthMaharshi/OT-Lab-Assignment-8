"""Surrogate objective used by greedy / ILP (a fast approximation of the simulator).

For a tower k = (cell, type) and enemy e (walking at base speed, no slows):
  dmg[k,e] = dps * (#ticks e spends in range of k)       -- what k could do to e alone
  cap[k]   = dps * (#ticks in which ANY enemy is in range of k)
             -- a tower fires at one target per tick, so total output is capped.

  f(S) = max  sum_{k,e} z[k,e]
         s.t. z[k,e] <= dmg[k,e],  sum_e z[k,e] <= cap[k],  sum_{k in S} z[k,e] <= hp_e

This is a max-flow value as a function of the source set S, hence monotone
submodular, so the greedy analysis applies to the SURROGATE. It still ignores
slow towers and the exact targeting order, so the guarantee says nothing about
the true simulator score. (An earlier version without cap[k] let a single gun
"kill" every enemy and badly mis-ranked layouts; see prompts/E/02_optimizers.md.)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


class Surrogate:
    def __init__(self, gmap, types, wave):
        self.cells = gmap.buildable()
        self.cell_idx = {c: i for i, c in enumerate(self.cells)}
        self.tnames = list(types)
        self.t_idx = {n: i for i, n in enumerate(self.tnames)}
        self.hp = np.array([e.hp for e in wave], dtype=float)
        L = len(gmap.path)
        px = np.array([p[0] for p in gmap.path], dtype=float)
        py = np.array([p[1] for p in gmap.path], dtype=float)
        C, T, E = len(self.cells), len(self.tnames), len(wave)
        self.dmg = np.zeros((C, T, E))
        self.cap = np.zeros((C, T))
        horizon = int(max(e.spawn + (L + 2) / e.speed for e in wave)) + 2
        for i, c in enumerate(self.cells):
            dist = np.hypot(px - c[0], py - c[1])
            for j, n in enumerate(self.tnames):
                tt = types[n]
                if tt.dps <= 0:
                    continue
                in_range = dist <= tt.range
                busy = np.zeros(horizon, dtype=bool)
                for e_idx, e in enumerate(wave):
                    m = np.arange(int((L + 2) / e.speed) + 1)
                    cell = (m * e.speed).astype(int)
                    ok = cell < L
                    ticks = m[ok][in_range[cell[ok]]]
                    self.dmg[i, j, e_idx] = tt.dps * len(ticks)
                    busy[e.spawn + ticks] = True
                self.cap[i, j] = tt.dps * busy.sum()

    def value_of(self, items):
        """items: list of (cell_idx, type_idx). Solves the small flow LP."""
        if not items:
            return 0.0
        K, E = len(items), len(self.hp)
        c = -np.ones(K * E)
        ub = np.concatenate([self.dmg[i, j] for i, j in items])
        A, b = [], []
        for k, (i, j) in enumerate(items):          # tower capacity
            row = np.zeros(K * E)
            row[k * E:(k + 1) * E] = 1
            A.append(row); b.append(self.cap[i, j])
        for e in range(E):                          # enemy hp
            row = np.zeros(K * E)
            row[e::E] = 1
            A.append(row); b.append(self.hp[e])
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                      bounds=list(zip([0] * (K * E), ub)), method="highs")
        return float(-res.fun)

    def value(self, placement):
        return self.value_of([(self.cell_idx[c], self.t_idx[n]) for c, n in placement.items()])
