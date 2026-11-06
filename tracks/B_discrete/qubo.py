"""QUBO and Ising tools, Max-Cut formulation, and classical solvers.

QUBO:   minimise  E(x) = x^T Q x,   x in {0,1}^n,  Q symmetric (diagonal = linear terms since x^2 = x).
Ising:  minimise  H(z) = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j,   z in {-1,+1}^n,   x = (1 - z)/2.
"""
from __future__ import annotations

import math

import numpy as np
from ortools.linear_solver import pywraplp


# ------------------------------------------------------------ construction
def qubo_from_terms(linear, pair=None):
    """linear: (n,) coefficients of x_i; pair: (n,n) upper-triangular coefficients of x_i x_j (i<j)."""
    Q = np.diag(np.asarray(linear, float))
    if pair is not None:
        B = np.triu(np.asarray(pair, float), 1)
        Q = Q + (B + B.T) / 2
    return Q


def energy(Q, x):
    x = np.asarray(x, float)
    return float(x @ Q @ x)


def qubo_to_ising(Q):
    """Return (h, J, const) with J strictly upper triangular, such that
    energy(Q, x) == const + h.z + sum_{i<j} J_ij z_i z_j for z = 1 - 2x."""
    d = np.diag(Q).copy()
    off = Q - np.diag(d)
    h = -d / 2 - off.sum(axis=1) / 2
    # x^T Q x holds each pair twice: 2*off_ij*x_i*x_j, and x_i*x_j = (1 - z_i - z_j + z_i*z_j)/4,
    # so the z_i*z_j coefficient is 2*off_ij/4 = off_ij/2.
    J = np.triu(off, 1) / 2
    const = d.sum() / 2 + off.sum() / 4
    return h, J, float(const)


def ising_energy(h, J, const, z):
    z = np.asarray(z, float)
    return float(const + h @ z + z @ J @ z)


def maxcut_qubo(n, edges):
    """minimise -cut(x) = sum_i -deg_i x_i + 2 sum_{(i,j) in E} x_i x_j."""
    Q = np.zeros((n, n))
    for i, j in edges:
        Q[i, i] -= 1
        Q[j, j] -= 1
        Q[i, j] += 1
        Q[j, i] += 1
    return Q


def cut_value(x, edges):
    x = np.asarray(x)
    return int(sum(x[i] != x[j] for i, j in edges))


def random_graph(n, p, seed=0):
    rng = np.random.default_rng(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


# ---------------------------------------------------------------- solvers
def _all_bits(n, lo, hi):
    m = np.arange(lo, hi)
    return ((m[:, None] >> np.arange(n)) & 1).astype(float)


def brute_force_qubo(Q, chunk=2 ** 16):
    """Exact minimum by enumerating all 2^n assignments (n <= ~24)."""
    n = len(Q)
    assert n <= 24
    best, bx = math.inf, None
    for lo in range(0, 2 ** n, chunk):
        X = _all_bits(n, lo, min(lo + chunk, 2 ** n))
        E = np.einsum("ij,ij->i", X @ Q, X)
        i = int(np.argmin(E))
        if E[i] < best:
            best, bx = float(E[i]), X[i].copy()
    return best, bx


def sa_qubo(Q, sweeps=1000, T0=2.0, T1=0.05, seed=0):
    """Single-flip simulated annealing with O(1) energy deltas."""
    rng = np.random.default_rng(seed)
    n = len(Q)
    d = np.diag(Q).copy()
    off = Q - np.diag(d)
    x = rng.integers(0, 2, n).astype(float)
    h = off @ x
    E = energy(Q, x)
    best, bx = E, x.copy()
    for s in range(sweeps):
        T = T0 * (T1 / T0) ** (s / max(1, sweeps - 1))
        u = rng.random(n)
        for k, i in enumerate(rng.permutation(n)):
            dE = (1 - 2 * x[i]) * (d[i] + 2 * h[i])
            if dE <= 0 or u[k] < math.exp(-dE / T):
                delta = 1 - 2 * x[i]
                x[i] += delta
                h += delta * off[:, i]
                E += dE
                if E < best - 1e-12:
                    best, bx = E, x.copy()
    return best, bx


def local_search_qubo(Q, restarts=20, seed=0):
    """Best-improvement 1-flip descent from random starts."""
    rng = np.random.default_rng(seed)
    n = len(Q)
    d = np.diag(Q).copy()
    off = Q - np.diag(d)
    best, bx = math.inf, None
    for _ in range(restarts):
        x = rng.integers(0, 2, n).astype(float)
        while True:
            dE = (1 - 2 * x) * (d + 2 * (off @ x))
            i = int(np.argmin(dE))
            if dE[i] >= -1e-12:
                break
            x[i] = 1 - x[i]
        E = energy(Q, x)
        if E < best:
            best, bx = E, x.copy()
    return best, bx


def ilp_maxcut(n, edges, time_limit_s=30):
    """Exact Max-Cut by ILP (edge variable y_e <= x_i + x_j and <= 2 - x_i - x_j)."""
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(int(time_limit_s * 1000))
    x = [s.BoolVar(f"x{i}") for i in range(n)]
    y = []
    for i, j in edges:
        e = s.NumVar(0, 1, "")
        s.Add(e <= x[i] + x[j])
        s.Add(e <= 2 - x[i] - x[j])
        y.append(e)
    s.Add(x[0] == 0)                                   # symmetry breaking
    s.Maximize(sum(y))
    status = s.Solve()
    xs = np.array([v.solution_value() > 0.5 for v in x]).astype(float)
    return cut_value(xs, edges), status == pywraplp.Solver.OPTIMAL, xs
