"""Top-level worker functions (must be importable for multiprocessing on Windows)."""
import numpy as np

_X = _y = None


def burn(n):
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


def burn_block(block):
    return [burn(n) for n in block]


def init_data(n, d, seed):
    global _X, _y
    rng = np.random.default_rng(seed)
    _X = rng.normal(size=(n, d))
    _y = (rng.random(n) < 0.5).astype(float)


def shard_grad(args):
    w, lo, hi = args
    X, y = _X[lo:hi], _y[lo:hi]
    p = 1 / (1 + np.exp(-(X @ w)))
    return X.T @ (p - y)
