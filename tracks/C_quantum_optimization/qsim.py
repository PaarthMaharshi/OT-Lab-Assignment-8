"""A small numpy quantum simulator written for this track (statevector + density matrix).

Qubit 0 is the most significant bit of the basis index, i.e. index = sum_q b_q 2^(n-1-q).
A circuit is a list of ops (kind, qubits, pidx, scale):
    kind  : 'h' 'x' 'cnot' 'rx' 'ry' 'rz' 'rzz'
    pidx  : index into the parameter vector theta (None for fixed gates)
    scale : angle = scale * theta[pidx]
Rotations are RX(a) = exp(-i a X/2), RZZ(a) = exp(-i a Z Z/2); all have generator
eigenvalues +-1/2, which is what makes the two-term parameter-shift rule exact.
"""
from __future__ import annotations

import numpy as np

H_ = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_ = np.array([[1, 0], [0, -1]], dtype=complex)


def gate_matrix(kind, a=0.0):
    c, s = np.cos(a / 2), np.sin(a / 2)
    if kind == "h": return H_
    if kind == "x": return X_
    if kind == "rx": return np.array([[c, -1j * s], [-1j * s, c]])
    if kind == "ry": return np.array([[c, -s], [s, c]], dtype=complex)
    if kind == "rz": return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])
    if kind == "cnot": return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    if kind == "rzz": return np.diag(np.exp(-1j * a / 2 * np.array([1, -1, -1, 1])))
    raise ValueError(kind)


# ------------------------------------------------------------------ statevector
def zero_state(n):
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1
    return psi


def apply_1q(psi, U, q):
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)


def apply_2q(psi, U4, a, b):
    U = U4.reshape(2, 2, 2, 2)
    return np.moveaxis(np.tensordot(U, psi, axes=([2, 3], [a, b])), (0, 1), (a, b))


def angle_of(op, theta, shifts=None, k=None):
    kind, qs, pidx, scale = op
    a = 0.0 if pidx is None else scale * theta[pidx]
    if shifts and k in shifts:
        a += shifts[k]
    return a


def run(ops, theta, n, shifts=None):
    """Return the final state as a flat vector. `shifts` = {op index: extra angle} (for parameter-shift)."""
    psi = zero_state(n)
    for k, op in enumerate(ops):
        kind, qs, _, _ = op
        U = gate_matrix(kind, angle_of(op, theta, shifts, k))
        psi = apply_1q(psi, U, qs[0]) if len(qs) == 1 else apply_2q(psi, U, qs[0], qs[1])
    return psi.reshape(-1)


def bit_table(n):
    idx = np.arange(2 ** n)
    return np.array([(idx >> (n - 1 - q)) & 1 for q in range(n)])      # shape (n, 2^n)


def maxcut_diag(n, edges):
    B = bit_table(n)
    return sum((B[i] != B[j]).astype(float) for i, j in edges)


def z_string_diag(n, qubits):
    B = bit_table(n)
    return np.prod([1 - 2 * B[q] for q in qubits], axis=0).astype(float)


def expval_diag(psi, d):
    return float(np.real(np.sum(np.abs(psi) ** 2 * d)))


def expval_dense(psi, M):
    return float(np.real(np.vdot(psi, M @ psi)))


def sample_mean(psi, d, shots, rng):
    """Estimate <diag> from `shots` projective measurements (finite-shot estimator)."""
    p = np.abs(psi) ** 2
    p = p / p.sum()
    return float(d[rng.choice(len(p), size=shots, p=p)].mean())


def sample_bitstrings(psi, shots, rng):
    p = np.abs(psi) ** 2
    return rng.choice(len(p), size=shots, p=p / p.sum())


# ------------------------------------------------------- parameter-shift gradient
def param_shift_grad(ops, theta, n, expval):
    """d<O>/d theta_i = sum over the gates using theta_i of scale * [f(a+pi/2) - f(a-pi/2)] / 2.
    `expval(psi) -> float`. Costs 2 circuit executions per parametrised gate."""
    g = np.zeros(len(theta))
    runs = 0
    for k, (kind, qs, pidx, scale) in enumerate(ops):
        if pidx is None:
            continue
        fp = expval(run(ops, theta, n, {k: np.pi / 2}))
        fm = expval(run(ops, theta, n, {k: -np.pi / 2}))
        g[pidx] += scale * 0.5 * (fp - fm)
        runs += 2
    return g, runs


def metric_tensor(ops, theta, n, eps=1e-6):
    """Fubini-Study metric g_ij = Re<di psi|dj psi> - Re(<di psi|psi><psi|dj psi>) (central differences
    of the simulated state; on hardware this needs extra circuits or a block-diagonal approximation)."""
    psi = run(ops, theta, n)
    D = []
    for i in range(len(theta)):
        e = np.zeros(len(theta)); e[i] = eps
        D.append((run(ops, theta + e, n) - run(ops, theta - e, n)) / (2 * eps))
    D = np.array(D)
    ov = D.conj() @ psi
    return np.real(D.conj() @ D.T) - np.real(np.outer(ov, ov.conj()))


# ----------------------------------------------------------------- density matrix
def _dm_unitary(rho, U, qs, n):
    if len(qs) == 1:
        q = qs[0]
        rho = np.moveaxis(np.tensordot(U, rho, axes=([1], [q])), 0, q)
        return np.moveaxis(np.tensordot(rho, U.conj(), axes=([q + n], [1])), -1, q + n)
    a, b = qs
    U4 = U.reshape(2, 2, 2, 2)
    rho = np.moveaxis(np.tensordot(U4, rho, axes=([2, 3], [a, b])), (0, 1), (a, b))
    return np.moveaxis(np.tensordot(rho, U4.conj(), axes=([a + n, b + n], [2, 3])), (-2, -1), (a + n, b + n))


def depolarize(rho, q, p, n):
    """Single-qubit depolarizing channel: rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)."""
    if p == 0:
        return rho
    acc = (1 - p) * rho
    for P in (X_, Y_, Z_):
        acc = acc + (p / 3) * _dm_unitary(rho, P, (q,), n)
    return acc


def run_dm(ops, theta, n, p_dep=0.0):
    """Density-matrix simulation with depolarizing noise after every gate (on each qubit it touched).
    Returns the diagonal (outcome probabilities)."""
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1
    for k, op in enumerate(ops):
        kind, qs, _, _ = op
        rho = _dm_unitary(rho, gate_matrix(kind, angle_of(op, theta)), qs, n)
        for q in qs:
            rho = depolarize(rho, q, p_dep, n)
    return np.real(np.diag(rho.reshape(2 ** n, 2 ** n)))
