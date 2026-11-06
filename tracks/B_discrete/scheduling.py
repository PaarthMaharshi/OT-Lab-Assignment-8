"""Weighted interval scheduling: dynamic programming vs greedy vs ILP.
Pick non-overlapping jobs (start, end, weight) to maximise total weight."""
from __future__ import annotations

import bisect

import numpy as np
from ortools.linear_solver import pywraplp


def make_jobs(n, seed=0, horizon=1000):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, horizon, n)
    e = s + rng.integers(5, horizon // 5, n)
    w = rng.integers(1, 100, n)
    return [(int(a), int(b), int(c)) for a, b, c in zip(s, e, w)]


def weighted_dp(jobs):
    """O(n log n). Jobs touching at an endpoint do not overlap (end == start is allowed)."""
    J = sorted(jobs, key=lambda j: j[1])
    ends = [j[1] for j in J]
    best = [0] * (len(J) + 1)
    for i, (s, e, w) in enumerate(J, 1):
        p = bisect.bisect_right(ends, s, 0, i - 1)      # last job finishing <= s
        best[i] = max(best[i - 1], best[p] + w)
    return best[-1]


def greedy_earliest_finish(jobs):
    """Optimal for UNWEIGHTED intervals, but ignores weights."""
    tot, last = 0, -1
    for s, e, w in sorted(jobs, key=lambda j: j[1]):
        if s >= last:
            tot += w
            last = e
    return tot


def greedy_max_weight(jobs):
    chosen = []
    for s, e, w in sorted(jobs, key=lambda j: -j[2]):
        if all(e <= cs or s >= ce for cs, ce, _ in chosen):
            chosen.append((s, e, w))
    return sum(j[2] for j in chosen)


def ilp(jobs):
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = [s.BoolVar(f"x{i}") for i in range(len(jobs))]
    for i in range(len(jobs)):
        for j in range(i + 1, len(jobs)):
            if jobs[i][0] < jobs[j][1] and jobs[j][0] < jobs[i][1]:
                s.Add(x[i] + x[j] <= 1)
    s.Maximize(sum(jobs[i][2] * x[i] for i in range(len(jobs))))
    s.Solve()
    return round(s.Objective().Value())


def brute_force(jobs):
    n = len(jobs)
    best = 0
    for m in range(2 ** n):
        ch = [jobs[i] for i in range(n) if m >> i & 1]
        if all(a[1] <= b[0] or b[1] <= a[0] for ai, a in enumerate(ch) for b in ch[ai + 1:]):
            best = max(best, sum(c[2] for c in ch))
    return best
