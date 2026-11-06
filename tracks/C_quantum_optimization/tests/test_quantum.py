"""Checks: slide values (PPT 3), analytic results, and a cross-check against PennyLane."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qopt                                   # noqa: E402
import qsim                                   # noqa: E402

EDGES4 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]          # slide 28/30 graph, max cut 4


def test_bell_state_slide16():
    ops = [("h", (0,), None, 0), ("cnot", (0, 1), None, 0)]
    p = np.abs(qsim.run(ops, [], 2)) ** 2
    assert p == pytest.approx([0.5, 0, 0, 0.5])


def test_hadamard_interference_slide12():
    psi = qsim.run([("h", (0,), None, 0), ("h", (0,), None, 0)], [], 1)
    assert np.abs(psi) ** 2 == pytest.approx([1, 0])


def test_born_rule_and_shots_slide6_7():
    rng = np.random.default_rng(0)
    psi = np.array([0.6, 0.8])
    z = np.array([1.0, -1.0])
    assert qsim.expval_diag(psi, z) == pytest.approx(0.36 - 0.64)
    est = [qsim.sample_mean(psi, z, 1000, rng) for _ in range(300)]
    assert np.std(est) == pytest.approx(2 * np.sqrt(0.36 * 0.64) / np.sqrt(1000), rel=0.15)   # 1/sqrt(shots)


def test_vqe_one_qubit_slide22_24():
    ops = [("ry", (0,), 0, 1.0)]
    f = lambda th: qsim.expval_diag(qsim.run(ops, th, 1), np.array([1.0, -1.0]))
    assert f([0.7]) == pytest.approx(np.cos(0.7))
    th = 0.5
    for expect in (0.740, 1.077):                              # slide: theta1 = 0.740, theta2 = 1.077
        g, _ = qsim.param_shift_grad(ops, np.array([th]), 1, lambda s: qsim.expval_diag(s, np.array([1.0, -1.0])))
        assert g[0] == pytest.approx(-np.sin(th))              # parameter-shift = analytic gradient
        th = th - 0.5 * g[0]
        assert th == pytest.approx(expect, abs=1e-3)
    g0, _ = qsim.param_shift_grad(ops, np.array([0.0]), 1, lambda s: qsim.expval_diag(s, np.array([1.0, -1.0])))
    assert g0[0] == pytest.approx(0)                           # stuck at theta = 0


def test_qnn_one_qubit_slide25():
    ops = [("ry", (0,), 0, 1.0), ("ry", (0,), 1, 1.0)]         # encode x then trainable theta
    psi = qsim.run(ops, np.array([0.4, 0.2]), 1)
    assert qsim.expval_diag(psi, np.array([1.0, -1.0])) == pytest.approx(np.cos(0.6)) and np.cos(0.6) == pytest.approx(0.825, abs=1e-3)


def test_param_shift_matches_finite_difference_on_random_circuit():
    rng = np.random.default_rng(1)
    ops = qopt.hea_ops(3, 2)
    th = rng.uniform(0, 2 * np.pi, qopt.n_params(ops))
    d = qsim.z_string_diag(3, [0, 1, 2])
    f = lambda t: qsim.expval_diag(qsim.run(ops, t, 3), d)
    g, runs = qsim.param_shift_grad(ops, th, 3, lambda s: qsim.expval_diag(s, d))
    fd = np.array([(f(th + e) - f(th - e)) / 2e-6 for e in np.eye(len(th)) * 1e-6])
    assert g == pytest.approx(fd, abs=1e-7) and runs == 2 * len(th)


def test_qaoa_shared_parameter_shift_is_exact():
    edges = [(0, 1), (1, 2), (0, 2)]
    ops = qopt.qaoa_ops(3, edges, 2)
    d = qsim.maxcut_diag(3, edges)
    th = np.array([0.3, 0.9, 0.4, 0.7])
    f = lambda t: qsim.expval_diag(qsim.run(ops, t, 3), d)
    g, _ = qsim.param_shift_grad(ops, th, 3, lambda s: qsim.expval_diag(s, d))
    fd = np.array([(f(th + e) - f(th - e)) / 2e-6 for e in np.eye(4) * 1e-6])
    assert g == pytest.approx(fd, abs=1e-6)


def test_fast_qaoa_equals_gate_level_circuit():
    edges = EDGES4
    d = qsim.maxcut_diag(4, edges)
    th = np.array([0.571, 0.8, 1.26, 0.3])
    a = qsim.run(qopt.qaoa_ops(4, edges, 2), th, 4)
    b = qopt.qaoa_state_fast(4, d, th[:2], th[2:])
    assert abs(np.vdot(a, b)) == pytest.approx(1.0)            # equal up to global phase


def test_qaoa_slide30_maxcut_values():
    d = qsim.maxcut_diag(4, EDGES4)
    assert d.max() == 4 and d.mean() == pytest.approx(2.5)      # random guessing = 2.5
    _, e1 = qopt.optimize_qaoa(4, d, 1, restarts=20)
    _, e2 = qopt.optimize_qaoa(4, d, 2, restarts=20)
    assert e1 == pytest.approx(3.237, abs=2e-3)                 # slide: p=1 E[cut] = 3.237
    assert e2 == pytest.approx(3.856, abs=2e-3)                 # slide: p=2 E[cut] = 3.856


def test_ising_ground_energy_and_vqe_slide28():
    d = -qsim.maxcut_diag(4, EDGES4)                            # H_C diagonal = -cut, ground energy -4
    ops = qopt.hea_ops(4, 2)
    assert qopt.n_params(ops) == 12                             # slide: 12-parameter ansatz
    from scipy.optimize import minimize
    f = lambda t: qsim.expval_diag(qsim.run(ops, t, 4), d)
    rng = np.random.default_rng(0)
    best = min(minimize(f, rng.uniform(0, 2 * np.pi, 12), method="BFGS").fun for _ in range(8))
    assert best == pytest.approx(-4.0, abs=1e-4)


def test_metric_tensor_single_qubit_ry():
    g = qsim.metric_tensor([("ry", (0,), 0, 1.0)], np.array([0.8]), 1)
    assert g[0, 0] == pytest.approx(0.25, abs=1e-6)             # Fubini-Study metric of RY is 1/4


def test_density_matrix_matches_statevector_without_noise():
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    ops = qopt.qaoa_ops(4, edges, 2)
    th = np.array([0.4, 0.9, 0.5, 0.2])
    assert qsim.run_dm(ops, th, 4, 0.0) == pytest.approx(np.abs(qsim.run(ops, th, 4)) ** 2, abs=1e-12)


def test_depolarizing_channel_analytic():
    ops = [("x", (0,), None, 0)]                                # |1>, <Z> = -1
    for p in (0.0, 0.1, 0.3):
        probs = qsim.run_dm(ops, [], 1, p)
        assert probs[0] - probs[1] == pytest.approx(-(1 - 4 * p / 3))       # <Z> after depolarizing
    assert qsim.run_dm(ops, [], 1, 0.75) == pytest.approx([0.5, 0.5])        # p = 3/4 -> maximally mixed


def test_crosscheck_against_pennylane():
    qml = pytest.importorskip("pennylane")
    n = 3
    ops = qopt.hea_ops(n, 2)
    th = np.random.default_rng(2).uniform(0, 2 * np.pi, qopt.n_params(ops))
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ(t):
        k = 0
        for l in range(3):
            for q in range(n):
                qml.RY(t[k], wires=q); k += 1
            if l < 2:
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))
    mine = qsim.expval_diag(qsim.run(ops, th, n), qsim.z_string_diag(n, [0, 2]))
    assert mine == pytest.approx(float(circ(th)), abs=1e-10)
