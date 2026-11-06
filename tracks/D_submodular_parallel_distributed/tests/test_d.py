import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dlib as d  # noqa: E402


def test_coverage_is_submodular_and_greedy_bound_holds():
    sets = d.make_coverage(30, 100, 0.1, 1)
    assert d.is_submodular_on_samples(lambda S: d.coverage(sets, S), 30) == 0
    for seed in range(8):
        sets = d.make_coverage(30, 100, 0.1, seed)
        for k in (3, 5):
            g = d.coverage(sets, d.greedy_cover(sets, k))
            assert g >= (1 - 1 / np.e) * d.opt_cover(sets, k) - 1e-9


def test_non_submodular_function_is_detected():
    assert d.is_submodular_on_samples(lambda S: len(S) ** 2, 20) > 0      # increasing returns


def test_slide_makespans():                                              # PPT2 slides 10-14
    q = [[8, 7], [6, 5], [4, 3], [2, 1]]
    assert d.makespan_static_queues(q) == 15
    assert d.makespan_dynamic([8, 7, 6, 5, 4, 3, 2, 1], 4) == 9
    assert d.makespan_work_stealing(q) == 11


def test_amdahl_gustafson_slides():
    assert d.amdahl(0.1, 16) == pytest.approx(6.4) and d.gustafson(0.1, 16) == pytest.approx(14.5)


def test_pipeline_bubble_formula_matches_simulation():
    for S, m in ((4, 8), (4, 32), (3, 5)):
        assert 1 - m / d.pipeline_makespan(S, m) == pytest.approx(d.pipeline_bubble(S, m))
    assert d.pipeline_bubble(4, 8) == pytest.approx(3 / 11)


def test_mgda_min_norm_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g1, g2 = rng.normal(size=5), rng.normal(size=5)
        a, _ = d.mgda_weights(g1, g2)
        grid = np.linspace(0, 1, 20001)
        best = grid[np.argmin([np.linalg.norm(t * g1 + (1 - t) * g2) for t in grid])]
        assert a == pytest.approx(best, abs=1e-3)


def test_fedavg_one_step_equals_centralized_gradient_step_when_iid_equal_sizes():
    clients, _ = d.make_clients(5, 100, 6, noniid=False, seed=2)
    w_fed, _ = d.fedavg(clients, 1, 1, 0.5)
    w_c, _ = d.centralized_gd(clients, 1, 0.5)
    assert w_fed == pytest.approx(w_c, abs=1e-12)


def test_multitask_gradients_match_finite_differences():
    p = d.TwoTask(5, 3, 60)
    x = p.flat(p.init(3))
    g1, g2 = p.grads(p.unflat(x))
    for j, g in enumerate((g1, g2)):
        fd = np.array([(p.losses(p.unflat(x + e))[j] - p.losses(p.unflat(x - e))[j]) / 2e-6 for e in np.eye(len(x)) * 1e-6])
        assert g == pytest.approx(fd, abs=1e-6)
