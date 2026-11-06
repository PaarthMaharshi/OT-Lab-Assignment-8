import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from td.maps import spiral_map  # noqa: E402
from td.qubo_tower import energies_all, tower_qubo  # noqa: E402
from td.sim import DEFAULT_TYPES, validate_placement  # noqa: E402


def test_spiral_path_is_a_connected_non_touching_path():
    g = spiral_map(13, 11)
    p = g.path
    assert len(set(p)) == len(p) > 60
    for a, b in zip(p, p[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1                       # consecutive cells adjacent
    for i, a in enumerate(p):                                                 # no cell touches a non-consecutive path cell
        for j in range(i + 2, len(p)):
            assert abs(a[0] - p[j][0]) + abs(a[1] - p[j][1]) > 1
    assert len(g.buildable()) > 20


def test_qubo_energy_matches_formula_and_penalty_enforces_k():
    g = spiral_map(9, 7)
    tt = DEFAULT_TYPES["gun"]
    cells = g.buildable()[:12]
    Q, const, cells, v = tower_qubo(g, tt, k=3, P=1000.0, cells=cells)
    E = energies_all(Q, const)
    best = int(np.argmin(E))
    x = [(best >> (len(cells) - 1 - i)) & 1 for i in range(len(cells))]
    assert sum(x) == 3                                                        # large penalty -> exactly k towers
    path = np.array(g.path, float)
    cover = np.array([np.hypot(path[:, 0] - c[0], path[:, 1] - c[1]) <= tt.range for c in cells], float)
    xs = np.array(x, float)
    direct = -(tt.dps * cover.sum(1) * xs).sum() + sum(tt.dps * (cover[i] @ cover[j]) * xs[i] * xs[j] for i in range(12) for j in range(i + 1, 12))
    assert E[best] == pytest.approx(direct)
    validate_placement(g, {cells[i]: "gun" for i in range(12) if x[i]}, DEFAULT_TYPES, 30)
