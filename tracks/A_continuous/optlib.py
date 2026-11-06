"""Continuous optimizers written from scratch (numpy only).

Every first-order optimizer has the signature  opt(grad, x0, steps, lr, ...) -> xs
where `grad(x)` may be deterministic (full batch) or stochastic (mini-batch), and
`xs` is the (steps+1, d) array of iterates. Update rules follow the lecture slides
(PPT 1, slides 8-19) so the slides' worked examples can be used as unit tests.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def _run(update, x0, steps):
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    for _ in range(steps):
        x = update(x)
        xs.append(x.copy())
    return np.array(xs)


def gd(grad, x0, steps, lr):
    """w <- w - lr * grad f(w). Full batch if grad is exact, SGD / mini-batch otherwise."""
    return _run(lambda x: x - lr * grad(x), x0, steps)


def momentum(grad, x0, steps, lr, beta=0.9):
    """v <- beta v + lr grad(w);  w <- w - v."""
    v = np.zeros_like(np.asarray(x0, float))

    def upd(x):
        nonlocal v
        v = beta * v + lr * grad(x)
        return x - v
    return _run(upd, x0, steps)


def nag(grad, x0, steps, lr, beta=0.9):
    """Nesterov: v <- beta v + lr grad(w - beta v);  w <- w - v."""
    v = np.zeros_like(np.asarray(x0, float))

    def upd(x):
        nonlocal v
        v = beta * v + lr * grad(x - beta * v)
        return x - v
    return _run(upd, x0, steps)


def adagrad(grad, x0, steps, lr, eps=1e-8):
    G = np.zeros_like(np.asarray(x0, float))

    def upd(x):
        nonlocal G
        g = grad(x)
        G = G + g * g
        return x - lr * g / (np.sqrt(G) + eps)
    return _run(upd, x0, steps)


def rmsprop(grad, x0, steps, lr, beta=0.9, eps=1e-8):
    E = np.zeros_like(np.asarray(x0, float))

    def upd(x):
        nonlocal E
        g = grad(x)
        E = beta * E + (1 - beta) * g * g
        return x - lr * g / (np.sqrt(E) + eps)
    return _run(upd, x0, steps)


def adam(grad, x0, steps, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = np.zeros_like(np.asarray(x0, float))
    v = np.zeros_like(m)
    t = 0

    def upd(x):
        nonlocal m, v, t
        t += 1
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        return x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return _run(upd, x0, steps)


def newton(grad, hess, x0, steps, lr=1.0, damping=0.0):
    """w <- w - lr (H + damping I)^-1 g. Heads to the NEAREST stationary point
    (minimum, maximum or saddle); it is not a descent method on non-convex f."""
    def upd(x):
        H = np.atleast_2d(hess(x))
        g = np.atleast_1d(grad(x))
        return x - lr * np.linalg.solve(H + damping * np.eye(len(g)), g)
    return _run(upd, x0, steps)


def newton_cg(grad, hvp, x0, steps, cg_iters=50, tol=1e-10):
    """Hessian-free Newton: solve H d = g by conjugate gradients using only
    Hessian-vector products. Returns (xs, total_hvp_calls)."""
    calls = 0
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    for _ in range(steps):
        def mv(v, x=x):
            nonlocal calls
            calls += 1
            return hvp(x, v)
        op = LinearOperator((len(x), len(x)), matvec=mv, dtype=float)
        d, _ = cg(op, grad(x), maxiter=cg_iters, rtol=tol)
        x = x - d
        xs.append(x.copy())
    return np.array(xs), calls


def gauss_newton(residual, jac, x0, steps, lm=0.0):
    """Minimise 1/2 ||r(x)||^2:  x <- x - (J^T J + lm I)^-1 J^T r.
    lm > 0 is Levenberg-Marquardt damping. Uses first derivatives only."""
    def upd(x):
        J = jac(x)
        return x - np.linalg.solve(J.T @ J + lm * np.eye(len(x)), J.T @ residual(x))
    return _run(upd, x0, steps)


def hvp_fd(grad, x, v, eps=1e-5):
    """Hessian-vector product by central differences of the gradient
    (two gradient calls, no Hessian formed)."""
    return (grad(x + eps * v) - grad(x - eps * v)) / (2 * eps)
