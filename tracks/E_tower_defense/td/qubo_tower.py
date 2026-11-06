"""QUBO formulation of tower placement (one tower type, k towers) and its exact / annealing / QAOA solutions.

E(x) = - sum_c v_c x_c + sum_{c<c'} o_cc' x_c x_c' + P (sum_c x_c - k)^2
  v_c   = dps * (#path cells in range of c)            value of a tower at c
  o_cc' = dps * (#path cells in range of both c, c')   overlap penalty (second-order inclusion-exclusion for coverage)
The cardinality term P(sum x - k)^2 enforces exactly k towers (budget = k * cost)."""
import numpy as np


def tower_qubo(gmap, tt, k, P, cells=None):
    cells = cells or gmap.buildable()
    path = np.array(gmap.path, float)
    cover = np.array([np.hypot(path[:, 0] - c[0], path[:, 1] - c[1]) <= tt.range for c in cells], float)     # (C, L)
    v = tt.dps * cover.sum(axis=1)
    O = tt.dps * (cover @ cover.T)
    n = len(cells)
    Q = np.zeros((n, n))
    Q[np.diag_indices(n)] = -v + P * (1 - 2 * k)
    off = (np.triu(O, 1) + 2 * P * np.triu(np.ones((n, n)), 1))     # pair coefficient of x_c x_c'
    Q += (off + off.T) / 2
    return Q, P * k * k, cells, v


def energies_all(Q, const=0.0):
    n = len(Q)
    idx = np.arange(2 ** n)
    X = ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(float)      # qubit 0 = most significant, matches qsim
    return np.einsum("ij,ij->i", X @ Q, X) + const
