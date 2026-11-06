"""Placement optimizers: greedy and ILP (on the surrogate), annealing (on the true simulator).

Problem: choose at most one tower type per buildable cell, total cost <= budget,
to maximize damage. Greedy/ILP optimize the surrogate (fast, but approximate);
simulated annealing optimizes the real simulator score directly (slow, exact).
"""
from __future__ import annotations

import math
import random

import numpy as np
from ortools.linear_solver import pywraplp

from .sim import simulate


# --------------------------------------------------------------------- greedy
def greedy(sur, types, budget):
    """Cost-benefit greedy on the surrogate, compared with the best single tower.

    Guarantee note: this is submodular maximization under a KNAPSACK constraint.
    The classic cardinality bound (1-1/e) does NOT apply; the known bound for
    'cost-benefit greedy vs best single item' is (1/2)(1-1/e) ~ 0.316.
    """
    cands = [(i, j) for i in range(len(sur.cells)) for j, n in enumerate(sur.tnames)
             if types[n].dps > 0 and types[n].cost <= budget]
    chosen, used, spent, cur = [], set(), 0.0, 0.0
    single_best, single_val = None, 0.0
    first = True
    while True:
        best, best_ratio, best_val = None, 1e-12, cur
        for (i, j) in cands:
            c = types[sur.tnames[j]].cost
            if i in used or spent + c > budget:
                continue
            v = sur.value_of(chosen + [(i, j)])
            if first and v > single_val:
                single_best, single_val = (i, j), v
            if (v - cur) / c > best_ratio:
                best, best_ratio, best_val = (i, j), (v - cur) / c, v
        first = False
        if best is None:
            break
        chosen.append(best); used.add(best[0]); spent += types[sur.tnames[best[1]]].cost
        cur = best_val
    if single_best is not None and single_val > cur:
        chosen = [single_best]
    return {sur.cells[i]: sur.tnames[j] for i, j in chosen}


# ------------------------------------------------------------------------ ILP
def ilp(sur, types, budget, time_limit_s=120):
    """Exact optimum of the surrogate flow model (x_k binary, z[k,e] continuous).

    Returns (placement, proven_optimal)."""
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(int(time_limit_s * 1000))
    C, T, E = sur.dmg.shape
    keys = [(i, j) for i in range(C) for j, n in enumerate(sur.tnames) if types[n].dps > 0]
    x = {k: s.BoolVar(f"x_{k[0]}_{k[1]}") for k in keys}
    z = {}
    for (i, j) in keys:
        for e in range(E):
            if sur.dmg[i, j, e] > 0:
                z[i, j, e] = s.NumVar(0, float(sur.dmg[i, j, e]), "")
                s.Add(z[i, j, e] <= float(sur.dmg[i, j, e]) * x[i, j])
    for (i, j) in keys:                                         # tower firing capacity
        s.Add(sum(z[i, j, e] for e in range(E) if (i, j, e) in z) <= float(sur.cap[i, j]) * x[i, j])
    for e in range(E):                                          # enemy hp
        s.Add(sum(v for (i, j, ee), v in z.items() if ee == e) <= float(sur.hp[e]))
    for i in range(C):                                          # one tower per cell
        s.Add(sum(x[i, j] for j in range(T) if (i, j) in x) <= 1)
    s.Add(sum(types[sur.tnames[j]].cost * v for (_, j), v in x.items()) <= budget)
    s.Maximize(sum(z.values()))
    status = s.Solve()
    placement = {}
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for (i, j), v in x.items():
            if v.solution_value() > 0.5:
                placement[sur.cells[i]] = sur.tnames[j]
    return placement, status == pywraplp.Solver.OPTIMAL


# ------------------------------------------------------------------ annealing
def _cost(p, types):
    return sum(types[n].cost for n in p.values())


def anneal(gmap, types, wave, budget, evals=800, seed=0, t0=150.0, t1=1.0,
           init=None):
    """Simulated annealing on the TRUE simulator score. `evals` = number of
    simulator calls (the fair-comparison budget). Returns (best_placement, best_score)."""
    rng = random.Random(seed)
    cells = gmap.buildable()
    names = list(types)

    def score(p):
        r = simulate(gmap, p, types, wave)
        return r.damage - 1e-3 * r.leaked_hp          # tie-break toward fewer leaked HP

    cur = dict(init) if init else {}
    spent = _cost(cur, types)
    # random start: fill greedily at random
    if not init:
        for c in rng.sample(cells, len(cells)):
            n = rng.choice(names)
            if spent + types[n].cost <= budget:
                cur[c] = n
                spent += types[n].cost
    cur_s = score(cur)
    best, best_s = dict(cur), cur_s
    for k in range(evals - 1):
        temp = t0 * (t1 / t0) ** (k / max(1, evals - 2))
        cand = dict(cur)
        move = rng.random()
        free = [c for c in cells if c not in cand]
        if cand and move < 0.35:                                   # relocate
            c = rng.choice(list(cand))
            n = cand.pop(c)
            cand[rng.choice(free + [c])] = n
        elif cand and move < 0.60:                                 # change type
            c = rng.choice(list(cand))
            cand[c] = rng.choice(names)
        elif move < 0.85 and free:                                 # add
            cand[rng.choice(free)] = rng.choice(names)
        elif cand:                                                 # remove
            cand.pop(rng.choice(list(cand)))
        if _cost(cand, types) > budget:
            continue                                               # infeasible: reject, no eval
        s = score(cand)
        if s >= cur_s or rng.random() < math.exp((s - cur_s) / temp):
            cur, cur_s = cand, s
            if s > best_s:
                best, best_s = dict(cand), s
    return best, best_s
