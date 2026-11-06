"""Quantum algorithms written gate by gate on the Track C numpy simulator (qubit 0 = most significant bit).
No library implementation of any algorithm is used; PennyLane/numpy are only used in tests as an independent check.

Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor (order finding for N=15) and adiabatic quantum annealing (simulated).
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_quantum_optimization"))
import qsim  # noqa: E402

H1, X1 = qsim.H_, qsim.X_


def zero(n):
    return qsim.zero_state(n)


def h(psi, q): return qsim.apply_1q(psi, H1, q)
def x(psi, q): return qsim.apply_1q(psi, X1, q)
def cnot(psi, c, t): return qsim.apply_2q(psi, qsim.gate_matrix("cnot"), c, t)


def cp(psi, a, b, theta):
    """Controlled phase: |11> -> e^{i theta}|11>."""
    return qsim.apply_2q(psi, np.diag([1, 1, 1, np.exp(1j * theta)]), a, b)


def swap(psi, a, b):
    U = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    return qsim.apply_2q(psi, U, a, b)


def probs(psi): return np.abs(psi.reshape(-1)) ** 2


def marginal(psi, n_total, qubits):
    """Probability distribution of the listed qubits (first listed = most significant)."""
    p = (np.abs(psi) ** 2).reshape((2,) * n_total)
    other = tuple(i for i in range(n_total) if i not in qubits)
    p = p.sum(axis=other) if other else p
    order = sorted(qubits)
    p = np.transpose(p, [order.index(q) for q in qubits])
    return p.reshape(-1)


# ------------------------------------------------------- Deutsch-Jozsa / Bernstein-Vazirani
def deutsch_jozsa(n, f_table):
    """f_table: array of 0/1 of length 2^n (constant or balanced). Phase-oracle version, ONE oracle call.
    Returns P(all-zero measurement on the n input qubits): 1 for constant f, 0 for balanced f."""
    psi = zero(n)
    for q in range(n):
        psi = h(psi, q)
    psi = (psi.reshape(-1) * (1 - 2 * np.asarray(f_table))).reshape((2,) * n)          # the single query
    for q in range(n):
        psi = h(psi, q)
    return float(probs(psi)[0])


def bernstein_vazirani(secret_bits):
    """Gate-level: f(x) = s.x mod 2 implemented with CNOTs onto an ancilla in |->. One query recovers s."""
    n = len(secret_bits)
    psi = zero(n + 1)
    psi = x(psi, n)
    for q in range(n + 1):
        psi = h(psi, q)
    for i, s in enumerate(secret_bits):                                                  # oracle U_f
        if s:
            psi = cnot(psi, i, n)
    for q in range(n):
        psi = h(psi, q)
    p = marginal(psi, n + 1, list(range(n)))
    idx = int(np.argmax(p))
    return [int(b) for b in format(idx, f"0{n}b")], float(p[idx])


# ------------------------------------------------------------------------------- Grover
def grover(n, marked, iterations):
    """Returns (P(marked), oracle_calls). The oracle is a sign flip of |marked>; the diffusion operator is H^n (2|0><0| - I) H^n."""
    psi = zero(n)
    for q in range(n):
        psi = h(psi, q)
    flat_sign = np.ones(2 ** n); flat_sign[marked] = -1
    zero_flip = np.ones(2 ** n); zero_flip[0] = -1
    for _ in range(iterations):
        psi = (psi.reshape(-1) * flat_sign).reshape((2,) * n)                            # oracle (1 query)
        for q in range(n): psi = h(psi, q)
        psi = (psi.reshape(-1) * zero_flip).reshape((2,) * n)
        for q in range(n): psi = h(psi, q)
    return float(probs(psi)[marked]), iterations


def grover_optimal_iterations(n):
    return int(math.floor(math.pi / 4 * math.sqrt(2 ** n)))


# --------------------------------------------------------------------------- QFT / QPE
def qft(psi, qubits, inverse=False):
    """QFT on the listed qubits (first = most significant). Inverse = reversed gate order with negated angles."""
    t = len(qubits)
    ops = []
    for j in range(t):
        ops.append(("h", qubits[j]))
        for k in range(j + 1, t):
            ops.append(("cp", qubits[j], qubits[k], math.pi / 2 ** (k - j)))
    for j in range(t // 2):
        ops.append(("swap", qubits[j], qubits[t - 1 - j]))
    if inverse:
        ops = ops[::-1]
    for op in ops:
        if op[0] == "h": psi = h(psi, op[1])
        elif op[0] == "swap": psi = swap(psi, op[1], op[2])
        else: psi = cp(psi, op[1], op[2], -op[3] if inverse else op[3])
    return psi


def qft_matrix(t):
    n = 2 ** t
    U = np.zeros((n, n), dtype=complex)
    for col in range(n):
        psi = zero(t).reshape(-1) * 0
        psi[col] = 1
        U[:, col] = qft(psi.reshape((2,) * t), list(range(t))).reshape(-1)
    return U


def qpe(phi, t):
    """Phase estimation of U = diag(1, e^{2 pi i phi}) with eigenstate |1>. Returns the distribution over the t-bit estimate."""
    n = t + 1
    psi = x(zero(n), t)
    for j in range(t):
        psi = h(psi, j)
    for j in range(t):                                                     # counting qubit j controls U^(2^(t-1-j))
        psi = cp(psi, j, t, 2 * math.pi * phi * 2 ** (t - 1 - j))
    psi = qft(psi, list(range(t)), inverse=True)
    return marginal(psi, n, list(range(t)))


# -------------------------------------------------------------------------------- Shor
def shor_order_finding(a, N=15, t=8, rng=None):
    """Quantum order finding via phase estimation of U|y> = |a y mod N> on a 4-qubit work register.
    The controlled modular multiplications are applied as exact permutations of the work register (NOT decomposed into gates),
    so no gate-count claim is made. Returns the distribution over the t-bit counting register."""
    w = 4
    psi = np.zeros((2 ** t, 2 ** w), dtype=complex)
    psi[:, 1] = 1 / math.sqrt(2 ** t)                                        # counting register in |+>^t, work register in |1>
    for j in range(t):
        mult = pow(a, 2 ** (t - 1 - j), N)
        perm = np.arange(2 ** w)
        for y in range(N):
            perm[y] = (y * mult) % N
        sel = ((np.arange(2 ** t) >> (t - 1 - j)) & 1).astype(bool)         # counting qubit j == 1
        new = psi.copy()
        tmp = np.zeros_like(psi[sel])
        tmp[:, perm] = psi[sel]
        new[sel] = tmp
        psi = new
    full = psi.reshape((2,) * (t + w))
    full = qft(full, list(range(t)), inverse=True)
    return marginal(full, t + w, list(range(t)))


def order_from_measurement(m, t, a, N=15):
    """Continued-fraction step: m / 2^t ~ s / r  ->  candidate order r, accepted only if a^r = 1 mod N.
    STRICT: no search over multiples of r (that would let a useless measurement such as m = 0 succeed by classical trial)."""
    r = Fraction(m, 2 ** t).limit_denominator(N).denominator
    return r if pow(a, r, N) == 1 else None


def classical_order(a, N=15):
    r, v = 1, a % N
    while v != 1:
        v = (v * a) % N
        r += 1
    return r


# ------------------------------------------------------------------ adiabatic annealing
def anneal(n, edges, T, steps=None):
    """Closed-system adiabatic evolution H(s) = -(1-s) sum X_i + s H_C (H_C diagonal = -cut), s = t/T, from |+>^n.
    Returns (probability of an optimal cut, minimum gap between the two lowest levels of the symmetric (bit-flip-even) sector along the schedule). Simulated, not a real annealer."""
    N = 2 ** n
    cut = qsim.maxcut_diag(n, edges)
    HC = np.diag(-cut)
    HX = np.zeros((N, N))
    for q in range(n):
        for s_ in range(N):
            HX[s_ ^ (1 << (n - 1 - q)), s_] -= 1.0
    steps = steps or int(max(50, 10 * T))
    dt = T / steps
    flip = np.arange(N) ^ (N - 1)                       # global bit-flip permutation P: H commutes with it and |+>^n is P-even
    psi = np.ones(N, dtype=complex) / math.sqrt(N)
    gap = 1e9
    for k in range(steps):
        s = (k + 0.5) / steps
        Hs = (1 - s) * HX + s * HC
        if k % max(1, steps // 40) == 0:
            ev, V = np.linalg.eigh(Hs)
            parity = np.einsum("ij,ij->j", V, V[flip, :])          # <v|P|v> = +1 for P-even eigenvectors
            even = ev[parity > 0.5]                                # the evolution never leaves this sector
            gap = min(gap, float(even[1] - even[0]))
        psi = expm(-1j * dt * Hs) @ psi
    return float(np.sum(np.abs(psi[cut == cut.max()]) ** 2)), float(gap)
