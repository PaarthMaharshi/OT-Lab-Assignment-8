"""Feature selection as discrete optimization: greedy forward / backward, exhaustive,
and a QUBO formulation with a cardinality penalty. `score(subset)` is any callable
mapping a tuple of feature indices to a number to MAXIMISE."""
from __future__ import annotations

import itertools

import numpy as np

from qubo import qubo_from_terms


class Counted:
    """Wrap a score function and count how many times (= model fits) it is called."""

    def __init__(self, f):
        self.f, self.calls, self.cache = f, 0, {}

    def __call__(self, S):
        S = tuple(sorted(S))
        if S not in self.cache:
            self.calls += 1
            self.cache[S] = self.f(S)
        return self.cache[S]


def n_fits_forward(d, k):
    """Greedy forward selection of k out of d features: d + (d-1) + ... + (d-k+1) fits (slide 21: 955)."""
    return sum(d - t for t in range(k))


def greedy_forward(score, d, k):
    S = ()
    for _ in range(k):
        cand = [j for j in range(d) if j not in S]
        S = tuple(sorted(S + (max(cand, key=lambda j: score(S + (j,))),)))
    return S


def backward_elimination(score, d, k):
    S = tuple(range(d))
    while len(S) > k:
        drop = max(S, key=lambda j: score(tuple(s for s in S if s != j)))     # drop what hurts least
        S = tuple(s for s in S if s != drop)
    return S


def exhaustive(score, d, k):
    return max(itertools.combinations(range(d), k), key=score)


def feature_qubo(X, y, k, alpha=0.5, P=1.0):
    """minimise  -sum_i r_i x_i + alpha * sum_{i<j} c_ij x_i x_j + P (sum_i x_i - k)^2
    r_i = |corr(feature i, label)| (relevance), c_ij = |corr(i, j)| (redundancy).
    Penalty expansion: P (sum x - k)^2 = P*sum_i (1-2k) x_i + 2P*sum_{i<j} x_i x_j + P k^2."""
    d = X.shape[1]
    r = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(d)])
    C = np.abs(np.corrcoef(X, rowvar=False))
    pair = np.triu(alpha * C + 2 * P, 1)
    linear = -r + P * (1 - 2 * k)
    return qubo_from_terms(linear, pair), P * k * k, r, C
