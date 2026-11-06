import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qalgs as qa  # noqa: E402
import qml  # noqa: E402


def test_deutsch_jozsa_one_query_decides_constant_vs_balanced():
    rng = np.random.default_rng(0)
    for n in range(1, 8):
        for c in (0, 1):
            assert qa.deutsch_jozsa(n, np.full(2 ** n, c)) == pytest.approx(1.0)
        for _ in range(5):
            f = np.zeros(2 ** n, int); f[rng.choice(2 ** n, 2 ** (n - 1), replace=False)] = 1
            assert qa.deutsch_jozsa(n, f) == pytest.approx(0.0, abs=1e-12)


def test_bernstein_vazirani_recovers_secret_with_certainty():
    rng = np.random.default_rng(1)
    for n in range(1, 9):
        s = [int(b) for b in rng.integers(0, 2, n)]
        got, p = qa.bernstein_vazirani(s)
        assert got == s and p == pytest.approx(1.0)


def test_grover_matches_closed_form_and_slide_n4():
    for n in range(2, 9):
        N = 2 ** n
        theta = math.asin(1 / math.sqrt(N))
        for k in range(0, 6):
            p, _ = qa.grover(n, 3 % N, k)
            assert p == pytest.approx(math.sin((2 * k + 1) * theta) ** 2, abs=1e-10)
    assert qa.grover(2, 1, 1)[0] == pytest.approx(1.0)                       # slide 14: N=4, one round finds it with certainty
    assert qa.grover(10, 5, qa.grover_optimal_iterations(10))[0] > 0.99


def test_qft_equals_dft_matrix():
    for t in (1, 2, 3, 4, 5):
        N = 2 ** t
        F = np.array([[np.exp(2j * np.pi * j * k / N) for k in range(N)] for j in range(N)]) / math.sqrt(N)
        assert np.allclose(qa.qft_matrix(t), F, atol=1e-12)


def test_inverse_qft_undoes_qft():
    psi = np.random.default_rng(0).normal(size=(2,) * 4) + 1j * np.random.default_rng(1).normal(size=(2,) * 4)
    psi /= np.linalg.norm(psi)
    back = qa.qft(qa.qft(psi, [0, 1, 2, 3]), [0, 1, 2, 3], inverse=True)
    assert np.allclose(back, psi)


def test_qpe_exact_and_inexact_phase():
    p = qa.qpe(1 / 8, 3)
    assert p[1] == pytest.approx(1.0)                                         # phi = 1/8 = 0.001b -> outcome 1
    p = qa.qpe(0.3, 6)
    assert int(np.argmax(p)) == 19 and p[19] > 4 / math.pi ** 2               # nearest 6-bit estimate, probability bound


def test_shor_finds_orders_and_factors_of_15():
    t = 8
    for a, r_true in ((2, 4), (4, 2), (7, 4), (8, 4), (11, 2), (13, 4)):
        assert qa.classical_order(a) == r_true
        p = qa.shor_order_finding(a, 15, t)
        assert p.sum() == pytest.approx(1.0)
        peaks = [m for m in np.argsort(-p)[:r_true]]                        # r peaks at multiples of 2^t / r
        for m in peaks:
            assert abs(m - round(m * r_true / 2 ** t) * 2 ** t / r_true) < 1.5
    # the continued-fraction step recovers r from a good measurement
    assert qa.order_from_measurement(64, 8, 7) == 4 and qa.order_from_measurement(192, 8, 7) == 4
    assert qa.order_from_measurement(0, 8, 7) is None and qa.order_from_measurement(128, 8, 7) is None      # uninformative / half-period outcomes fail


def test_annealing_reaches_ground_state_for_long_times_and_fails_for_short():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    p_short, _ = qa.anneal(4, edges, 0.2)
    p_long, gap = qa.anneal(4, edges, 40)
    assert p_long > 0.95 > p_short and gap > 1e-3          # gap to the first level above the degenerate ground pair


def test_vqc_gradient_is_exact_and_kernel_is_valid():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, math.pi, size=(20, 2)); y = np.where(rng.random(20) < 0.5, 1.0, -1.0)
    p = rng.uniform(0, 2 * math.pi, 8)
    loss, g, runs = qml.vqc_grad(X, y, p, 2)
    fd = np.array([(np.mean((qml.vqc_output(X, p + e, 2) - y) ** 2) - np.mean((qml.vqc_output(X, p - e, 2) - y) ** 2)) / 2e-6 for e in np.eye(8) * 1e-6])
    assert g == pytest.approx(fd, abs=1e-7) and runs == (2 * 8 + 1) * 20
    K = qml.kernel_matrix(X, X)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1) and np.linalg.eigvalsh(K).min() > -1e-9      # PSD, unit diagonal
