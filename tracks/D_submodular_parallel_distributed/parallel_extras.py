"""Track D supplement (PPT 2 slides 25-28, PPT 4 slides 8-18). Run: python parallel_extras.py -> results/parallel_extras.json
  * communication cost T = alpha + n / beta, measured between two processes and fitted; batching small messages
  * local SGD / synchronisation frequency (communication rounds vs accuracy) - see FedAvg in experiments.py
  * top-k gradient sparsification with and without error feedback (simulated data-parallel SGD)
  * parallel hyper-parameter search: grid vs random vs successive halving, run on a process pool
  * planning formulas: model-memory fit, ring all-reduce volume/time, pipeline bubble, hybrid layouts, Amdahl for training"""
import itertools
import json
import os
import statistics as st
import time
from multiprocessing import Pipe, Pool, Process

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = {}


# ------------------------------------------------------------------ planning formulas (tested)
def training_memory_gb(params, bytes_per_param=16):          # fp32 weights 4 + grads 4 + Adam m,v 8 (slide 8)
    return params * bytes_per_param / 1e9


def ring_allreduce_bytes_per_worker(model_bytes, p):          # 2 (p-1)/p * model size (slide 11)
    return 2 * (p - 1) / p * model_bytes


def hybrid_layouts(gpus):                                       # ordered (d, t, p) with d*t*p = gpus
    return sum(1 for d in range(1, gpus + 1) for t in range(1, gpus + 1) if gpus % (d * t) == 0)


def amdahl(s, p):
    return 1 / (s + (1 - s) / p)


def successive_halving_cost(n_configs, eta, rounds):          # epoch units, budget x eta per round (slide 18)
    cost, n, b = 0, n_configs, 1
    for _ in range(rounds):
        cost += n * b
        n, b = n // eta, b * eta
    return cost


# ------------------------------------------------------------------ communication cost model
def _echo(conn):
    while True:
        msg = conn.recv_bytes()
        if msg == b"stop":
            return
        conn.send_bytes(b"k")


def exp_comm():
    a, b = Pipe()
    pr = Process(target=_echo, args=(b,))
    pr.start()
    sizes = [16, 1_000, 10_000, 100_000, 1_000_000, 4_000_000]
    times = {}
    for n in sizes:
        msg = b"x" * n
        ts = []
        for _ in range(30 if n < 1_000_000 else 10):
            t = time.perf_counter(); a.send_bytes(msg); a.recv_bytes(); ts.append(time.perf_counter() - t)
        times[n] = st.median(ts)
    # batching: 100 messages of 100 B vs one of 10 KB
    small = b"x" * 100
    t = time.perf_counter()
    for _ in range(100):
        a.send_bytes(small); a.recv_bytes()
    t_many = time.perf_counter() - t
    t = time.perf_counter(); a.send_bytes(b"x" * 10_000); a.recv_bytes(); t_one = time.perf_counter() - t
    a.send_bytes(b"stop"); pr.join()
    # alpha from the smallest message; 1/beta from the slope between the two largest (a plain least-squares line is
    # dominated by the large messages and gave a negative alpha in the first version)
    alpha = times[16]
    beta_inv = (times[4_000_000] - times[1_000_000]) / 3_000_000
    pred = {str(k): alpha + k * beta_inv for k in sizes}
    OUT["communication_cost"] = {"measured_round_trip_s": {str(k): v for k, v in times.items()}, "model_alpha_plus_n_over_beta_s": pred,
                                 "alpha_us": alpha * 1e6, "bandwidth_MB_per_s": 1 / beta_inv / 1e6, "batching_100x100B_s": t_many, "one_10KB_message_s": t_one,
                                 "batching_speedup": t_many / t_one, "note": "Python multiprocessing Pipe on one machine (round trip includes a 1-byte reply)"}


# ------------------------------------------------------------------ top-k sparsification
def exp_topk():
    rng = np.random.default_rng(0)
    n, d, workers = 4000, 200, 8
    X = rng.normal(size=(n, d)) * (rng.random(d) ** 2 + 0.05)
    w_true = rng.normal(size=d)
    y = (X @ w_true + 0.5 * rng.normal(size=n) > 0).astype(float)
    shards = np.array_split(np.arange(n), workers)

    def loss(w):
        z = X @ w
        return float(np.mean(np.logaddexp(0, z) - y * z))

    def run(keep, feedback, steps=300, lr=0.5):
        w = np.zeros(d); mem = [np.zeros(d) for _ in range(workers)]; sent = 0
        for _ in range(steps):
            agg = np.zeros(d)
            for k, idx in enumerate(shards):
                g = X[idx].T @ (1 / (1 + np.exp(-(X[idx] @ w))) - y[idx]) / len(idx)
                if feedback:
                    g = g + mem[k]
                m = max(1, int(keep * d))
                top = np.argpartition(-np.abs(g), m - 1)[:m]
                c = np.zeros(d); c[top] = g[top]
                if feedback:
                    mem[k] = g - c                                  # keep what was not sent for later
                agg += c; sent += 2 * m                             # value + index per entry
            w -= lr * agg / workers
        return loss(w), sent / (steps * workers * d)
    full, _ = run(1.0, False)
    rows = {"no compression": {"final_loss": full, "fraction_of_dense_volume": 1.0}}
    for keep in (0.1, 0.01):
        for fb in (False, True):
            l, vol = run(keep, fb)
            rows[f"top-{int(keep * 100)}% {'with' if fb else 'without'} error feedback"] = {"final_loss": l, "fraction_of_dense_volume": vol / 1.0}
    OUT["topk_sparsification"] = {"setup": "8 simulated workers, logistic regression d=200, 300 synchronous steps", "rows": rows}


# ------------------------------------------------------------------ parallel hyper-parameter search
DATA = None


def _init():
    global DATA
    X, y = make_classification(4000, 30, n_informative=10, random_state=0)
    DATA = train_test_split(X, y, test_size=0.3, random_state=0)


def _trial(args):
    alpha, eta0, epochs = args
    Xtr, Xva, ytr, yva = DATA
    m = SGDClassifier(alpha=alpha, eta0=eta0, learning_rate="constant", max_iter=epochs, tol=None, random_state=0)
    m.fit(Xtr, ytr)
    return m.score(Xva, yva)


def exp_hpo():
    alphas = np.logspace(-6, -1, 9); etas = np.logspace(-4, 0, 9)
    grid = [(a, e) for a in alphas for e in etas]                 # 81 configurations
    rng = np.random.default_rng(0)
    rand = [(10 ** rng.uniform(-6, -1), 10 ** rng.uniform(-4, 0)) for _ in range(27)]
    out = {}
    with Pool(1, initializer=_init) as p1, Pool(8, initializer=_init) as p8:
        for pool in (p1, p8):                                     # warm-up: workers import sklearn and build the data first
            pool.map(_trial, [(1e-4, 0.01, 1)] * 16)
        for name, pool in (("1 process", p1), ("8 processes", p8)):
            t = time.perf_counter(); acc = pool.map(_trial, [(a, e, 27) for a, e in grid]); out[f"grid 81 x 27 epochs, {name}"] = {"seconds": time.perf_counter() - t, "best_val_acc": max(acc), "epoch_units": 81 * 27}
        t = time.perf_counter(); acc = p8.map(_trial, [(a, e, 27) for a, e in rand]); out["random 27 x 27 epochs, 8 processes"] = {"seconds": time.perf_counter() - t, "best_val_acc": max(acc), "epoch_units": 27 * 27}
        # successive halving on the random configurations: 27 -> 9 -> 3 -> 1, budget 1, 3, 9, 27 epochs
        t = time.perf_counter(); cfg = list(rand); budget = 1; units = 0
        while True:
            acc = p8.map(_trial, [(a, e, budget) for a, e in cfg]); units += len(cfg) * budget
            order = np.argsort(acc)[::-1]
            if len(cfg) == 1:
                break
            cfg = [cfg[i] for i in order[:max(1, len(cfg) // 3)]]; budget *= 3
        out["successive halving 27 configs, 8 processes"] = {"seconds": time.perf_counter() - t, "best_val_acc": float(acc[0]), "epoch_units": units}
    OUT["parallel_hyperparameter_search"] = out


def exp_formulas():
    OUT["planning_formulas"] = {
        "training_memory_GB_1B_params": training_memory_gb(1e9), "training_memory_GB_10B_params": training_memory_gb(1e10),
        "fits_80GB_GPU_1B": training_memory_gb(1e9) <= 80, "fits_80GB_GPU_10B": training_memory_gb(1e10) <= 80,
        "ring_allreduce_MB_per_worker_400MB_p8": ring_allreduce_bytes_per_worker(400, 8), "ring_allreduce_ms_at_10GBps": ring_allreduce_bytes_per_worker(400, 8) / 10_000 * 1000,
        "hybrid_layouts_64_gpus": hybrid_layouts(64), "amdahl_training_s0.1_p8": amdahl(0.1, 8), "successive_halving_epoch_units_27_eta3": successive_halving_cost(27, 3, 4),
        "note": "slide values: 16 GB / 160 GB, 700 MB, 70 ms, 28 layouts, 4.71x, 108 epoch units"}


if __name__ == "__main__":
    for f in (exp_formulas, exp_comm, exp_topk, exp_hpo):
        t = time.perf_counter(); f(); print(f.__name__, f"{time.perf_counter() - t:.1f}s", flush=True)
    lines = open(os.path.join(RES, "locality_sync.txt")).read().split("\n") if os.path.exists(os.path.join(RES, "locality_sync.txt")) else []
    OUT["locality_sync_openmp"] = {l.split()[0]: float(l.split()[1]) for l in lines if len(l.split()) == 2}
    json.dump(OUT, open(os.path.join(RES, "parallel_extras.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in OUT.items() if k != "locality_sync_openmp"}, indent=1, default=str))
