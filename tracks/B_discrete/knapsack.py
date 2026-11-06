"""0/1 knapsack: greedy, dynamic programming, branch-and-bound, ILP, brute force.

max sum v_i x_i   s.t.  sum w_i x_i <= cap,  x_i in {0,1}   (integer weights)
"""
from __future__ import annotations

import sys

import numpy as np
from ortools.linear_solver import pywraplp

sys.setrecursionlimit(10000)


def make_instance(n, kind="uncorrelated", seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(10, 101, size=n)
    v = rng.integers(10, 101, size=n) if kind == "uncorrelated" else w + 10   # strongly correlated: hard
    return v.astype(np.int64), w.astype(np.int64), int(w.sum() // 2)


def value_of(v, sel):
    return int(np.asarray(v)[np.asarray(sel, bool)].sum())


def feasible(w, sel, cap):
    return int(np.asarray(w)[np.asarray(sel, bool)].sum()) <= cap


def greedy_ratio(v, w, cap):
    """Take items by decreasing value/weight while they fit; return the better of
    that and the best single item (the classic 1/2-approximation)."""
    n = len(v)
    sel = np.zeros(n, bool)
    room = cap
    for i in np.argsort(-(v / w), kind="stable"):
        if w[i] <= room:
            sel[i] = True
            room -= w[i]
    fits = np.where(w <= cap)[0]
    if len(fits) and v[fits].max() > v[sel].sum():
        sel = np.zeros(n, bool)
        sel[fits[np.argmax(v[fits])]] = True
    return sel


def dp(v, w, cap):
    """Exact, O(n * cap) time and memory (pseudo-polynomial)."""
    n = len(v)
    T = np.zeros((n + 1, cap + 1), dtype=np.int64)
    for i in range(1, n + 1):
        T[i] = T[i - 1]
        wi = int(w[i - 1])
        if wi <= cap:
            T[i, wi:] = np.maximum(T[i - 1, wi:], T[i - 1, :cap + 1 - wi] + v[i - 1])
    sel = np.zeros(n, bool)
    c = cap
    for i in range(n, 0, -1):
        if T[i, c] != T[i - 1, c]:
            sel[i - 1] = True
            c -= int(w[i - 1])
    return sel


def branch_and_bound(v, w, cap, node_limit=1_500_000):
    """Depth-first B&B with the fractional (LP) relaxation as upper bound.
    Returns (selection, nodes_explored, proven_optimal)."""
    n = len(v)
    order = np.argsort(-(v / w), kind="stable")
    vs, ws = v[order].astype(float), w[order].astype(float)
    state = {"best": 0.0, "sel": np.zeros(n, bool), "nodes": 0, "hit": False}
    cur = np.zeros(n, bool)

    def bound(i, val, room):
        b = val
        for j in range(i, n):
            if ws[j] <= room:
                room -= ws[j]
                b += vs[j]
            else:
                return b + vs[j] * room / ws[j]
        return b

    def rec(i, val, room):
        state["nodes"] += 1
        if state["nodes"] > node_limit:
            state["hit"] = True
            return
        if val > state["best"]:
            state["best"], state["sel"] = val, cur.copy()
        if i == n or bound(i, val, room) < state["best"] + 1:      # integer values
            return
        if ws[i] <= room:
            cur[i] = True
            rec(i + 1, val + vs[i], room - ws[i])
            cur[i] = False
            if state["hit"]:
                return
        rec(i + 1, val, room)

    rec(0, 0.0, float(cap))
    sel = np.zeros(n, bool)
    sel[order[state["sel"]]] = True
    return sel, state["nodes"], not state["hit"]


def ilp(v, w, cap, time_limit_s=60):
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(int(time_limit_s * 1000))
    x = [s.BoolVar(f"x{i}") for i in range(len(v))]
    s.Add(sum(int(w[i]) * x[i] for i in range(len(v))) <= cap)
    s.Maximize(sum(int(v[i]) * x[i] for i in range(len(v))))
    status = s.Solve()
    sel = np.array([xi.solution_value() > 0.5 for xi in x])
    return sel, status == pywraplp.Solver.OPTIMAL


def brute_force(v, w, cap):
    n = len(v)
    assert n <= 22
    best, bsel = -1, None
    for lo in range(0, 2 ** n, 2 ** 16):
        m = np.arange(lo, min(lo + 2 ** 16, 2 ** n))
        bits = ((m[:, None] >> np.arange(n)) & 1).astype(np.int64)
        ok = bits @ w <= cap
        vals = np.where(ok, bits @ v, -1)
        i = int(np.argmax(vals))
        if vals[i] > best:
            best, bsel = int(vals[i]), bits[i].astype(bool)
    return bsel
