"""Correctness checks. Expected numbers come from the lecture slides (PPT 1) or from
hand derivation / finite differences, NOT from the code under test."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optlib import (adagrad, adam, gd, gauss_newton, hvp_fd, momentum, nag,   # noqa: E402
                    newton, newton_cg, rmsprop)
from problems import (DoubleWell, LogCosh, Logistic, MLP, NonConvexBounded,   # noqa: E402
                      Quadratic, Rosenbrock, Saddle)


# ----------------------------------------------------- slide worked examples
def test_gd_slide8():                       # f=(w-3)^2, w0=0, lr=0.1 -> 0.6, 1.08, 1.464
    xs = gd(lambda x: 2 * (x - 3), [0.0], 3, 0.1)[:, 0]
    assert xs[1:] == pytest.approx([0.6, 1.08, 1.464])


def test_lr_regimes_slide9():               # factor (1-2 lr): 1.0 bounces 0<->6, 1.1 diverges
    g = lambda x: 2 * (x - 3)
    assert gd(g, [0.0], 4, 1.0)[:, 0] == pytest.approx([0, 6, 0, 6, 0])
    assert abs(gd(g, [0.0], 30, 1.1)[-1, 0] - 3) > 100


def test_momentum_slide14():                # f=w^2, w0=5, lr=.1, beta=.9 -> 4, 2.3
    xs = momentum(lambda x: 2 * x, [5.0], 2, 0.1, 0.9)[:, 0]
    assert xs[1:] == pytest.approx([4.0, 2.3])


def test_nag_slide15():                     # -> 4, 2.48
    xs = nag(lambda x: 2 * x, [5.0], 2, 0.1, 0.9)[:, 0]
    assert xs[1:] == pytest.approx([4.0, 2.48])


def test_adam_slide16():                    # f=w^2, w0=3, lr=.1 -> w1 = 2.9
    assert adam(lambda x: 2 * x, [3.0], 1, 0.1)[1, 0] == pytest.approx(2.9, abs=1e-6)


def test_adagrad_slide17():                 # w0=3, lr=1 -> 2, 1.445
    xs = adagrad(lambda x: 2 * x, [3.0], 2, 1.0, eps=0)[:, 0]
    assert xs[1:] == pytest.approx([2.0, 1.4453], abs=1e-3)


def test_rmsprop_slide18():                 # w0=3, lr=.1, beta=.9 -> 2.684
    assert rmsprop(lambda x: 2 * x, [3.0], 1, 0.1, 0.9, eps=0)[1, 0] == pytest.approx(2.6838, abs=1e-3)


def test_newton_slide19():                  # f=w^4-3w^2+w, w0=2 -> 1.5
    p = DoubleWell()
    assert newton(p.grad, p.hess, [2.0], 1)[1, 0] == pytest.approx(1.5)


# ----------------------------------------------------- derivative correctness
def fd_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


@pytest.mark.parametrize("prob,x", [
    (Quadratic(10, 5), np.arange(1.0, 6.0)), (Rosenbrock(), np.array([-1.2, 1.0])),
    (DoubleWell(), np.array([0.7])), (Saddle(), np.array([0.3, 0.8])),
    (LogCosh(), np.array([0.5, -1.0])), (NonConvexBounded(), np.array([0.4, 1.5]))])
def test_analytic_gradients(prob, x):
    assert prob.grad(x) == pytest.approx(fd_grad(prob.f, x), abs=1e-5)


def test_logistic_gradient_and_hessian():
    p = Logistic(n=200, d=6, lam=1e-2)
    w = np.random.default_rng(0).normal(size=6)
    assert p.grad(w) == pytest.approx(fd_grad(p.f, w), abs=1e-6)
    H = p.hess(w)
    Hfd = np.array([fd_grad(lambda u, i=i: p.grad(u)[i], w) for i in range(6)])
    assert H == pytest.approx(Hfd, abs=1e-5)
    v = np.random.default_rng(1).normal(size=6)
    assert p.hvp(w, v) == pytest.approx(H @ v)                 # matrix-free HVP is exact
    assert hvp_fd(p.grad, w, v) == pytest.approx(H @ v, abs=1e-6)


def test_backprop_matches_finite_differences():
    m = MLP(h=5, lam=1e-2, seed=1, n=50)
    th = np.random.default_rng(3).normal(size=m.dim) * 0.5
    g = m.grad(th)
    gfd = fd_grad(m.loss, th, 1e-6)
    rel = np.abs(g - gfd).max() / (np.abs(gfd).max() + 1e-12)
    assert rel < 1e-6


# ----------------------------------------------------------- landscapes
def test_double_well_local_vs_global():
    p = DoubleWell()
    right = gd(p.grad, [2.0], 2000, 0.01)[-1, 0]
    left = gd(p.grad, [-2.0], 2000, 0.01)[-1, 0]
    assert right == pytest.approx(1.1309, abs=1e-3) and p.f([right]) > p.f([left]) + 1
    assert left == pytest.approx(-1.3009, abs=1e-3)


def test_saddle_gd_stalls_and_newton_goes_to_saddle():
    p = Saddle()
    ev = np.linalg.eigvalsh(p.hess(np.zeros(2)))
    assert ev.min() < 0 < ev.max()                               # indefinite = saddle
    assert gd(p.grad, [1.0, 0.0], 500, 0.1)[-1] == pytest.approx([0, 0], abs=1e-6)   # stalls
    assert abs(gd(p.grad, [1.0, 1e-3], 500, 0.1)[-1, 1]) == pytest.approx(np.sqrt(2), abs=1e-3)
    assert newton(p.grad, p.hess, [1.0, 0.5], 30)[-1] == pytest.approx([0, 0], abs=1e-6)


# ----------------------------------------------------------- convergence theory
def test_gd_strongly_convex_linear_rate():
    q = Quadratic(10, 8)
    x0 = np.ones(8)
    xs = gd(q.grad, x0, 60, 1 / q.L)
    for k in range(61):
        assert q.f(xs[k]) <= (1 - q.mu / q.L) ** (2 * k) * q.f(x0) + 1e-12


def test_gd_convex_sublinear_bound():
    p, x0 = LogCosh(), np.array([3.0, -2.0, 1.5])
    xs = gd(p.grad, x0, 300, 1 / p.L)
    for k in range(1, 301):                                      # f_k - f* <= L |x0-x*|^2 / (2k)
        assert p.f(xs[k]) - 0 <= p.L * (x0 @ x0) / (2 * k) + 1e-12


def test_gd_nonconvex_gradient_norm_bound():
    p, x0 = NonConvexBounded(), np.array([0.3, -1.2, 2.0])
    K = 400
    xs = gd(p.grad, x0, K, 1 / p.L)
    best = min(np.linalg.norm(p.grad(x)) for x in xs[:K])
    assert best <= np.sqrt(2 * p.L * (p.f(x0) - 0) / K) + 1e-12


# ------------------------------------------------------- second order methods
def test_newton_solves_quadratic_in_one_step():
    q = Quadratic(100, 6)
    assert newton(q.grad, q.hess, np.ones(6), 1)[-1] == pytest.approx(np.zeros(6), abs=1e-12)


def test_newton_cg_matches_newton():
    p = Logistic(n=300, d=8, lam=1e-2)
    a = newton(p.grad, p.hess, np.zeros(8), 8)[-1]
    b, calls = newton_cg(p.grad, p.hvp, np.zeros(8), 8)
    assert b[-1] == pytest.approx(a, abs=1e-7) and calls > 0


def test_gauss_newton_exact_fit():
    t = np.linspace(0, 2, 30)
    a, b = 2.0, -0.7
    y = a * np.exp(b * t)
    res = lambda x: x[0] * np.exp(x[1] * t) - y
    jac = lambda x: np.stack([np.exp(x[1] * t), x[0] * t * np.exp(x[1] * t)], axis=1)
    x = gauss_newton(res, jac, [1.5, -0.3], 15)[-1]
    assert x == pytest.approx([a, b], abs=1e-8)


def test_fisher_equals_hessian_only_at_optimum_for_empirical():
    p = Logistic(n=1000, d=6, lam=0.0, seed=2)
    w_far = 3 * p.wstar            # NOT w=0: there p=1/2 for every sample and the two coincide exactly
    Ht = p.fisher_true(p.wstar)
    assert p.hess(p.wstar) == pytest.approx(Ht, abs=1e-4)       # true Fisher == Hessian (canonical link)
    far = np.linalg.norm(p.fisher_empirical(w_far) - p.hess(w_far)) / np.linalg.norm(p.hess(w_far))
    near = np.linalg.norm(p.fisher_empirical(p.wstar) - p.hess(p.wstar)) / np.linalg.norm(p.hess(p.wstar))
    assert near < far
