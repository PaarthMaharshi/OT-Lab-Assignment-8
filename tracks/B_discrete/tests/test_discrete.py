"""Checks use the lecture slides' worked examples (PPT 1 slides 24-27, PPT 3 slides 23, 28)
and brute-force enumeration as independent ground truth."""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import featsel                       # noqa: E402
import knapsack as kn                # noqa: E402
import qubo as qb                    # noqa: E402
import scheduling as sc              # noqa: E402

# slide 24/25 illustrative accuracies; features A,B,C,D = 0,1,2,3
ACC = {(0,): .70, (1,): .75, (2,): .72, (3,): .68, (0, 1): .80, (0, 2): .74, (0, 3): .73,
       (1, 2): .78, (1, 3): .76, (2, 3): .84, (0, 1, 2, 3): .83, (0, 1, 2): .82,
       (0, 1, 3): .81, (0, 2, 3): .85, (1, 2, 3): .80}
score = lambda S: ACC[tuple(sorted(S))]


def test_slide24_greedy_misses_optimum():
    assert featsel.greedy_forward(score, 4, 2) == (0, 1)               # {A,B} = 0.80
    assert featsel.exhaustive(score, 4, 2) == (2, 3)                   # {C,D} = 0.84


def test_slide25_backward_elimination_finds_it():
    assert featsel.backward_elimination(score, 4, 2) == (2, 3)


def test_slide21_fit_count_955():
    assert featsel.n_fits_forward(100, 10) == 955
    c = featsel.Counted(lambda S: len(S) + sum(S) * 1e-6)
    featsel.greedy_forward(c, 20, 5)
    assert c.calls == featsel.n_fits_forward(20, 5)


def test_slide26_knapsack_example():
    v, w = np.array([5, 4, 3, 2]), np.array([3, 2, 2, 1])
    for sel in (kn.dp(v, w, 3), kn.branch_and_bound(v, w, 3)[0], kn.ilp(v, w, 3)[0], kn.brute_force(v, w, 3)):
        assert kn.value_of(v, sel) == 6 and list(np.where(sel)[0]) == [1, 3]      # items B and D
    assert kn.value_of(v, kn.greedy_ratio(v, w, 3)) in (5, 6)


@pytest.mark.parametrize("kind", ["uncorrelated", "strongly_correlated"])
def test_exact_knapsack_methods_agree_with_brute_force(kind):
    for seed in range(6):
        v, w, cap = kn.make_instance(14, kind, seed)
        opt = kn.value_of(v, kn.brute_force(v, w, cap))
        d, (b, nodes, proven), (i, ilp_opt) = kn.dp(v, w, cap), kn.branch_and_bound(v, w, cap), kn.ilp(v, w, cap)
        assert proven and ilp_opt
        for sel in (d, b, i):
            assert kn.feasible(w, sel, cap) and kn.value_of(v, sel) == opt
        g = kn.greedy_ratio(v, w, cap)
        assert kn.feasible(w, g, cap) and kn.value_of(v, g) >= opt / 2          # 1/2-approximation


def test_slide28_qubo_maxcut():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    Q = qb.maxcut_qubo(4, edges)
    assert np.diag(Q).tolist() == [-3, -2, -3, -2]
    best, x = qb.brute_force_qubo(Q)
    assert best == -4
    assert {tuple(x.astype(int))} <= {(0, 1, 0, 1), (1, 0, 1, 0)}
    assert qb.cut_value(x, edges) == 4


def test_qubo_ising_energies_match_for_every_assignment():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    Q = (A + A.T) / 2
    h, J, c = qb.qubo_to_ising(Q)
    for bits in itertools.product([0, 1], repeat=6):
        x = np.array(bits, float)
        assert qb.energy(Q, x) == pytest.approx(qb.ising_energy(h, J, c, 1 - 2 * x))


def test_slide28_ising_form_of_maxcut():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    h, J, c = qb.qubo_to_ising(qb.maxcut_qubo(4, edges))
    assert np.allclose(h, 0) and c == pytest.approx(-2.5)                # H = 1/2 sum ZZ - 5/2
    assert all(J[i, j] == pytest.approx(0.5) for i, j in edges if i < j)
    assert J[3, 0] == 0 and J[0, 3] == pytest.approx(0.5)


def test_solvers_find_known_maxcut_optimum():
    for seed in range(4):
        n = 12
        edges = qb.random_graph(n, 0.5, seed)
        Q = qb.maxcut_qubo(n, edges)
        opt, _ = qb.brute_force_qubo(Q)
        cut, proven, xs = qb.ilp_maxcut(n, edges)
        assert proven and cut == -opt
        e_sa, x_sa = qb.sa_qubo(Q, sweeps=300, seed=seed)
        assert e_sa == pytest.approx(opt) and qb.cut_value(x_sa, edges) == -opt
        e_ls, _ = qb.local_search_qubo(Q, restarts=30, seed=seed)
        assert e_ls >= opt - 1e-9                                            # a heuristic can never beat the optimum


def test_cardinality_penalty_expansion():
    # with no relevance/redundancy the QUBO energy must equal P (|S|-k)^2 for every x
    n, k, P = 7, 3, 2.5
    Q, const, _, _ = featsel.feature_qubo(np.random.default_rng(0).normal(size=(50, n)), np.random.default_rng(1).normal(size=50), k, alpha=0.0, P=P)
    X = np.random.default_rng(0).normal(size=(50, n)); y = np.random.default_rng(1).normal(size=50)
    r = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n)])
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, float)
        assert qb.energy(Q, x) + const == pytest.approx(-r @ x + P * (x.sum() - k) ** 2)


def test_weighted_intervals_dp_matches_brute_force_and_ilp():
    for seed in range(8):
        jobs = sc.make_jobs(12, seed, horizon=100)
        opt = sc.brute_force(jobs)
        assert sc.weighted_dp(jobs) == opt == sc.ilp(jobs)
        assert sc.greedy_earliest_finish(jobs) <= opt and sc.greedy_max_weight(jobs) <= opt


def test_lasso_soft_threshold_slide27():
    soft = lambda z, lam: np.sign(z) * max(abs(z) - lam, 0)
    assert soft(1.5, 0.5) == 1.0 and soft(0.3, 0.5) == 0
