"""Quantum machine learning on a simulator: a variational classifier with data re-uploading (trained with
parameter-shift gradients) and a fidelity quantum kernel, to be compared with classical models on the same data."""
from __future__ import annotations

import time

import numpy as np

import qalgs
import qsim

Z0 = np.array([1.0, 1.0, -1.0, -1.0])                       # <Z> of qubit 0 on a 2-qubit basis index (qubit 0 = MSB)
CNOT = qsim.gate_matrix("cnot")


def _rx_batch(x):
    c, s = np.cos(x / 2), np.sin(x / 2)
    U = np.zeros((len(x), 2, 2), dtype=complex)
    U[:, 0, 0] = U[:, 1, 1] = c
    U[:, 0, 1] = U[:, 1, 0] = -1j * s
    return U


def _fixed_layer(theta):
    """RZ(phi) RY(theta) on each of the two qubits, then CNOT. theta = [ry0, rz0, ry1, rz1]."""
    q0 = qsim.gate_matrix("rz", theta[1]) @ qsim.gate_matrix("ry", theta[0])
    q1 = qsim.gate_matrix("rz", theta[3]) @ qsim.gate_matrix("ry", theta[2])
    return CNOT @ np.kron(q0, q1)


def vqc_output(X, params, layers):
    """<Z_0> for every sample. Each layer: encode (RX(x1), RX(x2)), then trainable rotations + CNOT (data re-uploading)."""
    B = len(X)
    psi = np.zeros((B, 4), dtype=complex)
    psi[:, 0] = 1
    for l in range(layers):
        A, Bm = _rx_batch(X[:, 0]), _rx_batch(X[:, 1])
        E = np.einsum("bij,bkl->bikjl", A, Bm).reshape(B, 4, 4)
        psi = np.einsum("bij,bj->bi", E, psi)
        psi = psi @ _fixed_layer(params[4 * l:4 * l + 4]).T
    return (np.abs(psi) ** 2) @ Z0


def vqc_grad(X, y, params, layers):
    """Gradient of the mean-squared loss by the parameter-shift rule. Returns (loss, grad, circuit_executions)."""
    out = vqc_output(X, params, layers)
    loss = float(np.mean((out - y) ** 2))
    g = np.zeros_like(params)
    for i in range(len(params)):
        e = np.zeros_like(params); e[i] = np.pi / 2
        d_out = 0.5 * (vqc_output(X, params + e, layers) - vqc_output(X, params - e, layers))
        g[i] = np.mean(2 * (out - y) * d_out)
    return loss, g, (2 * len(params) + 1) * len(X)


def train_vqc(Xtr, ytr, layers=3, steps=60, lr=0.1, seed=0):
    """Adam on parameter-shift gradients. ytr in {-1,+1}. Returns (params, seconds, circuit_executions)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 2 * np.pi, 4 * layers)
    m = np.zeros_like(p); v = np.zeros_like(p); runs = 0
    t0 = time.perf_counter()
    for t in range(1, steps + 1):
        _, g, r = vqc_grad(Xtr, ytr, p, layers)
        runs += r
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        p = p - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return p, time.perf_counter() - t0, runs


def vqc_accuracy(X, y, params, layers):
    return float(np.mean(np.sign(vqc_output(X, params, layers)) == y))


# ---------------------------------------------------------------- quantum kernel
def feature_state(x, reps=2):
    """ZZ-type feature map on 2 qubits: H, RZ(2 x_i), RZZ(2 x_1 x_2), repeated."""
    psi = qalgs.zero(2)
    for _ in range(reps):
        psi = qalgs.h(psi, 0); psi = qalgs.h(psi, 1)
        psi = qsim.apply_1q(psi, qsim.gate_matrix("rz", 2 * x[0]), 0)
        psi = qsim.apply_1q(psi, qsim.gate_matrix("rz", 2 * x[1]), 1)
        psi = qsim.apply_2q(psi, qsim.gate_matrix("rzz", 2 * x[0] * x[1]), 0, 1)
    return psi.reshape(-1)


def kernel_matrix(XA, XB, reps=2):
    A = np.array([feature_state(x, reps) for x in XA])
    B = A if XB is XA else np.array([feature_state(x, reps) for x in XB])
    return np.abs(A.conj() @ B.T) ** 2                       # |<phi(x)|phi(x')>|^2
