"""Test objectives: analytic landscapes, logistic regression, and a small MLP with
hand-written backpropagation."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.datasets import make_moons


# ----------------------------------------------------------- analytic landscapes
class Quadratic:
    """f = 1/2 x^T A x, A = diag(1 .. kappa). Strongly convex, mu=1, L=kappa."""

    def __init__(self, kappa=100.0, d=20):
        self.a = np.logspace(0, np.log10(kappa), d)
        self.mu, self.L, self.fstar = 1.0, kappa, 0.0

    def f(self, x): return 0.5 * float(self.a @ (x * x))
    def grad(self, x): return self.a * x
    def hess(self, x): return np.diag(self.a)


class Rosenbrock:
    """Non-convex valley, global minimum 0 at (1, 1)."""
    fstar = 0.0

    def f(self, x): return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

    def grad(self, x):
        return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                         200 * (x[1] - x[0] ** 2)])

    def hess(self, x):
        return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                         [-400 * x[0], 200.0]])


class DoubleWell:
    """f(w) = w^4 - 3 w^2 + w  (slide 19). Local minimum ~1.13, global minimum ~-1.30."""

    def f(self, x): return float(x[0] ** 4 - 3 * x[0] ** 2 + x[0])
    def grad(self, x): return np.array([4 * x[0] ** 3 - 6 * x[0] + 1])
    def hess(self, x): return np.array([[12 * x[0] ** 2 - 6]])


class Saddle:
    """f = x^2 - y^2 + y^4/4. Saddle at (0,0); minima at (0, +-sqrt 2), f = -1."""

    def f(self, x): return float(x[0] ** 2 - x[1] ** 2 + x[1] ** 4 / 4)
    def grad(self, x): return np.array([2 * x[0], -2 * x[1] + x[1] ** 3])
    def hess(self, x): return np.array([[2.0, 0.0], [0.0, -2 + 3 * x[1] ** 2]])


class LogCosh:
    """Convex, L = 1, but NOT strongly convex (curvature -> 0 away from the minimum)."""
    fstar, L = 0.0, 1.0

    def f(self, x): return float(np.sum(np.log(np.cosh(x))))
    def grad(self, x): return np.tanh(x)
    def hess(self, x): return np.diag(1 / np.cosh(x) ** 2)


class NonConvexBounded:
    """f = sum x^2/(1+x^2). Smooth (L = 2), bounded below by 0, non-convex."""
    fstar, L = 0.0, 2.0

    def f(self, x): return float(np.sum(x ** 2 / (1 + x ** 2)))
    def grad(self, x): return 2 * x / (1 + x ** 2) ** 2


# ------------------------------------------------------------ logistic regression
class Logistic:
    """L2-regularised logistic regression, labels in {-1,+1}.
    f(w) = mean log(1+exp(-y x.w)) + lam/2 ||w||^2.  Strongly convex for lam > 0."""

    def __init__(self, n=2000, d=20, lam=1e-3, seed=0, decay=0.85):
        rng = np.random.default_rng(seed)
        scale = decay ** np.arange(d)                     # decaying feature scales -> low-rank-ish Hessian
        self.X = rng.normal(size=(n, d)) * scale
        w_true = rng.normal(size=d) / scale ** 0.5
        p = expit(self.X @ w_true)
        self.y = np.where(rng.random(n) < p, 1.0, -1.0)
        self.n, self.d, self.lam = n, d, lam
        self.rng = np.random.default_rng(seed + 1000)
        w = np.zeros(d)                                   # reference optimum by Newton
        for _ in range(60):
            w = w - np.linalg.solve(self.hess(w), self.grad(w))
        self.wstar, self.fstar = w, self.f(w)
        self.L = float(np.linalg.eigvalsh(self.hess(self.wstar)).max()) + 0.25 * 0   # local estimate
        self.L_global = 0.25 * np.linalg.eigvalsh(self.X.T @ self.X / n).max() + lam
        self.mu = lam

    def f(self, w):
        return float(np.mean(np.logaddexp(0, -self.y * (self.X @ w))) + 0.5 * self.lam * w @ w)

    def _gi(self, w, idx):
        z = self.y[idx] * (self.X[idx] @ w)
        coef = -self.y[idx] * expit(-z)
        return self.X[idx].T @ coef / len(coef) + self.lam * w

    def grad(self, w): return self._gi(w, slice(None))

    def batch_grad(self, bs):
        """Return a stochastic gradient function drawing a fresh mini-batch each call."""
        def g(w):
            idx = self.rng.integers(0, self.n, size=bs)
            return self._gi(w, idx)
        return g

    def hess(self, w):
        p = expit(self.X @ w)
        return (self.X.T * (p * (1 - p))) @ self.X / self.n + self.lam * np.eye(self.d)

    def hvp(self, w, v):
        """Exact matrix-free Hessian-vector product, O(n d)."""
        p = expit(self.X @ w)
        return self.X.T @ ((p * (1 - p)) * (self.X @ v)) / self.n + self.lam * v

    def fisher_true(self, w):
        """Expected Fisher of the logistic model = X^T diag(p(1-p)) X / n (no lam term)."""
        p = expit(self.X @ w)
        return (self.X.T * (p * (1 - p))) @ self.X / self.n

    def fisher_empirical(self, w):
        """Empirical Fisher = mean of g_i g_i^T using the observed labels."""
        y01 = (self.y + 1) / 2
        p = expit(self.X @ w)
        G = (p - y01)[:, None] * self.X
        return G.T @ G / self.n


# -------------------------------------------------------- MLP with manual backprop
class MLP:
    """2 -> h -> 1 network: tanh hidden layer, sigmoid output, cross-entropy + L2.
    Parameters live in one flat vector so every optimizer in optlib can train it."""

    def __init__(self, h=16, lam=0.0, seed=0, n=400, noise=0.2):
        X, y = make_moons(n_samples=2 * n, noise=noise, random_state=seed)
        self.Xtr, self.ytr = X[:n], y[:n].astype(float)
        self.Xte, self.yte = X[n:], y[n:].astype(float)
        self.h, self.lam = h, lam
        self.dim = 2 * h + h + h + 1
        rng = np.random.default_rng(seed)
        self.theta0 = np.concatenate([rng.normal(size=2 * h) * 0.5, np.zeros(h),
                                      rng.normal(size=h) * 0.5, [0.0]])
        self.rng = np.random.default_rng(seed + 7)

    def unpack(self, th):
        h = self.h
        return (th[:2 * h].reshape(2, h), th[2 * h:3 * h], th[3 * h:4 * h], th[4 * h])

    def forward(self, th, X):
        W1, b1, W2, b2 = self.unpack(th)
        A = np.tanh(X @ W1 + b1)
        return A, expit(A @ W2 + b2)

    def loss(self, th, X=None, y=None):
        X = self.Xtr if X is None else X
        y = self.ytr if y is None else y
        _, p = self.forward(th, X)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        W1, _, W2, _ = self.unpack(th)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                     + 0.5 * self.lam * (np.sum(W1 ** 2) + np.sum(W2 ** 2)))

    def grad(self, th, idx=None):
        """Backpropagation: forward pass stores activations, backward pass applies
        the chain rule layer by layer (PPT 1, slides 11-12)."""
        X = self.Xtr if idx is None else self.Xtr[idx]
        y = self.ytr if idx is None else self.ytr[idx]
        W1, b1, W2, b2 = self.unpack(th)
        A, p = self.forward(th, X)
        m = len(y)
        dz2 = (p - y) / m                                  # dL/d(output pre-activation)
        gW2 = A.T @ dz2 + self.lam * W2
        gb2 = dz2.sum()
        dA = np.outer(dz2, W2)                             # back through the output layer
        dz1 = dA * (1 - A ** 2)                            # tanh' = 1 - tanh^2
        gW1 = X.T @ dz1 + self.lam * W1
        gb1 = dz1.sum(axis=0)
        return np.concatenate([gW1.ravel(), gb1, gW2, [gb2]])

    def batch_grad(self, bs):
        def g(th):
            return self.grad(th, self.rng.integers(0, len(self.ytr), size=bs))
        return g

    def accuracy(self, th, test=True):
        X, y = (self.Xte, self.yte) if test else (self.Xtr, self.ytr)
        _, p = self.forward(th, X)
        return float(np.mean((p > 0.5) == (y > 0.5)))
