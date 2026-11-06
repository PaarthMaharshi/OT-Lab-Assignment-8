"""Circuits (QAOA, hardware-efficient ansatz), Hamiltonians and quantum-aware optimizers
(Adam with parameter-shift gradients, SPSA, quantum natural gradient)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import qsim


# ------------------------------------------------------------------- circuits
def qaoa_ops(n, edges, p):
    """|+>^n, then p times: cost layer exp(-i gamma_l C) and mixer exp(-i beta_l sum X).
    theta = [gamma_1..gamma_p, beta_1..beta_p]."""
    ops = [("h", (q,), None, 0.0) for q in range(n)]
    for l in range(p):
        ops += [("rzz", (i, j), l, -1.0) for i, j in edges]      # RZZ(-gamma) == exp(-i gamma cut) up to a global phase
        ops += [("rx", (q,), p + l, 2.0) for q in range(n)]      # RX(2 beta)
    return ops


def hea_ops(n, layers):
    """Hardware-efficient ansatz: (layers+1) RY layers separated by CNOT ladders; n*(layers+1) parameters."""
    ops, k = [], 0
    for l in range(layers + 1):
        for q in range(n):
            ops.append(("ry", (q,), k, 1.0)); k += 1
        if l < layers:
            ops += [("cnot", (q, q + 1), None, 0.0) for q in range(n - 1)]
    return ops


def n_params(ops):
    return 1 + max(o[2] for o in ops if o[2] is not None)


def qaoa_state_fast(n, cut_diag, gammas, betas):
    """Same state as run(qaoa_ops(...)) up to a global phase, using a diagonal cost layer."""
    psi = np.full((2,) * n, 1 / np.sqrt(2 ** n), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = (psi.reshape(-1) * np.exp(-1j * g * cut_diag)).reshape((2,) * n)
        U = qsim.gate_matrix("rx", 2 * b)
        for q in range(n):
            psi = qsim.apply_1q(psi, U, q)
    return psi.reshape(-1)


def tfim_dense(n, g=1.0):
    """Open transverse-field Ising chain H = -sum Z_i Z_{i+1} - g sum X_i (dense 2^n x 2^n)."""
    def op(P, q):
        mats = [np.eye(2, dtype=complex)] * n
        mats = list(mats); mats[q] = P
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n - 1):
        H -= op(qsim.Z_, q) @ op(qsim.Z_, q + 1)
    for q in range(n):
        H -= g * op(qsim.X_, q)
    return H


# -------------------------------------------------------------- QAOA optimizer
def optimize_qaoa(n, cut_diag, p, restarts=8, seed=0):
    """Maximise the exact expected cut over (gamma, beta) with L-BFGS from random starts."""
    rng = np.random.default_rng(seed)
    f = lambda th: -qsim.expval_diag(qaoa_state_fast(n, cut_diag, th[:p], th[p:]), cut_diag)
    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        r = minimize(f, x0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


# ------------------------------------------------------------------ optimizers
def adam_ps(fun, grad, theta0, steps, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    """Adam on parameter-shift gradients. `grad(theta) -> (g, circuits_used)`. Returns (theta, history)."""
    th = np.array(theta0, float); m = np.zeros_like(th); v = np.zeros_like(th)
    hist, runs = [], 0
    for t in range(1, steps + 1):
        g, r = grad(th); runs += r
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        th = th - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        hist.append((runs, fun(th)))
    return th, hist


def gd_ps(fun, grad, theta0, steps, lr=0.1):
    th = np.array(theta0, float); hist, runs = [], 0
    for _ in range(steps):
        g, r = grad(th); runs += r
        th = th - lr * g
        hist.append((runs, fun(th)))
    return th, hist


def spsa(fun, theta0, steps, a=0.2, c=0.1, A=10, alpha=0.602, gamma=0.101, seed=0):
    """Simultaneous perturbation: 2 evaluations per step whatever the dimension."""
    rng = np.random.default_rng(seed)
    th = np.array(theta0, float); hist, runs = [], 0
    for k in range(steps):
        ak, ck = a / (k + 1 + A) ** alpha, c / (k + 1) ** gamma
        d = rng.choice([-1.0, 1.0], size=len(th))
        g = (fun(th + ck * d) - fun(th - ck * d)) / (2 * ck) * d
        runs += 2
        th = th - ak * g
        hist.append((runs, fun(th)))
    return th, hist


def qng(fun, grad, metric, theta0, steps, lr=0.1, reg=1e-3, metric_cost=0):
    """Quantum natural gradient: theta <- theta - lr (g + reg I)^-1 grad."""
    th = np.array(theta0, float); hist, runs = [], 0
    for _ in range(steps):
        g, r = grad(th); runs += r + metric_cost
        G = metric(th)
        th = th - lr * np.linalg.solve(G + reg * np.eye(len(th)), g)
        hist.append((runs, fun(th)))
    return th, hist
