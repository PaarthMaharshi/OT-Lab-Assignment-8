"""Track D library: submodular greedy, scheduling/work-stealing simulation, pipeline model,
FedAvg and multi-task gradient weighting (MGDA, GradNorm)."""
from __future__ import annotations

import heapq

import numpy as np
from ortools.linear_solver import pywraplp


# ------------------------------------------------------------- submodular coverage
def make_coverage(n_sets=40, universe=150, density=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return [set(np.where(rng.random(universe) < density)[0].tolist()) for _ in range(n_sets)]


def coverage(sets, S):
    return len(set().union(*[sets[i] for i in S])) if S else 0


def greedy_cover(sets, k, f=None):
    f = f or (lambda S: coverage(sets, S))
    S = []
    for _ in range(k):
        rest = [i for i in range(len(sets)) if i not in S]
        S.append(max(rest, key=lambda i: f(S + [i])))
    return S


def opt_cover(sets, k):
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = [s.BoolVar("") for _ in sets]
    U = sorted(set().union(*sets))
    y = {e: s.BoolVar("") for e in U}
    for e in U:
        s.Add(y[e] <= sum(x[i] for i, A in enumerate(sets) if e in A))
    s.Add(sum(x) <= k)
    s.Maximize(sum(y.values()))
    s.Solve()
    return round(s.Objective().Value())


def is_submodular_on_samples(f, n, trials=300, seed=0):
    """Return the number of violations of f(A+x)-f(A) >= f(B+x)-f(B) for random A subset B."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        B = list(rng.choice(n, size=rng.integers(2, n // 2), replace=False))
        A = B[:rng.integers(1, len(B))]
        rest = [i for i in range(n) if i not in B]
        x = int(rng.choice(rest))
        if f(A + [x]) - f(A) < f(B + [x]) - f(B) - 1e-12:
            bad += 1
    return bad


# ----------------------------------------------------------------- scheduling model
def makespan_static(times, p):
    """Contiguous equal-size blocks assigned before running (OpenMP schedule(static))."""
    blocks = np.array_split(np.arange(len(times)), p)
    return max(sum(times[i] for i in b) for b in blocks)


def makespan_dynamic(times, p, order=None):
    """Task pool: the next task goes to the least-loaded worker (schedule(dynamic)); order=LPT sorts descending."""
    idx = list(order) if order is not None else range(len(times))
    h = [0.0] * p
    heapq.heapify(h)
    for i in idx:
        heapq.heappush(h, heapq.heappop(h) + times[i])
    return max(h)


def makespan_static_queues(queues):
    return max(sum(q) for q in queues)


def makespan_work_stealing(queues):
    """Event simulation. Each worker runs its own queue front-to-back; an idle worker steals the LAST
    unstarted task of the queue with the most remaining work. Only unstarted tasks can be stolen."""
    queues = [list(q) for q in queues]
    p = len(queues)
    free = [0.0] * p
    t_done = 0.0
    events = [(0.0, w) for w in range(p)]
    heapq.heapify(events)
    while events:
        t, w = heapq.heappop(events)
        if queues[w]:
            task = queues[w].pop(0)
        else:
            victims = [v for v in range(p) if queues[v]]
            if not victims:
                continue
            v = max(victims, key=lambda v: sum(queues[v]))
            task = queues[v].pop()
        fin = t + task
        t_done = max(t_done, fin)
        heapq.heappush(events, (fin, w))
    return t_done


def amdahl(s, p): return 1 / (s + (1 - s) / p)
def gustafson(s, p): return s + p * (1 - s)


def pipeline_makespan(stages, micro, t_stage=1.0):
    """GPipe-style forward schedule: stage s starts micro-batch m when it is free and stage s-1 finished m."""
    done = np.zeros((stages, micro))
    for m in range(micro):
        for s in range(stages):
            start = max(done[s, m - 1] if m else 0.0, done[s - 1, m] if s else 0.0)
            done[s, m] = start + t_stage
    return done[-1, -1]


def pipeline_bubble(stages, micro):
    return (stages - 1) / (micro + stages - 1)


# --------------------------------------------------------------------- FedAvg
def make_clients(n_clients=10, per_client=200, d=10, noniid=True, seed=0):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)
    clients = []
    for c in range(n_clients):
        shift = rng.normal(size=d) * (1.5 if noniid else 0.0)          # feature shift per client
        X = rng.normal(size=(per_client, d)) + shift
        p = 1 / (1 + np.exp(-(X @ w_true)))
        y = (rng.random(per_client) < p).astype(float)
        if noniid:                                                     # label skew: drop most of one class
            keep = (y == (c % 2)) | (rng.random(per_client) < 0.25)
            X, y = X[keep], y[keep]
        clients.append((X, y))
    return clients, w_true


def lr_loss(w, X, y):
    z = X @ w
    return float(np.mean(np.logaddexp(0, z) - y * z))


def lr_grad(w, X, y):
    return X.T @ (1 / (1 + np.exp(-(X @ w))) - y) / len(y)


def global_loss(w, clients):
    X = np.vstack([c[0] for c in clients]); y = np.concatenate([c[1] for c in clients])
    return lr_loss(w, X, y)


def fedavg(clients, rounds, local_steps, lr, batch=None, seed=0):
    """FedAvg: every round each client runs `local_steps` SGD steps from the global model; the server
    averages the results weighted by client size. Returns the loss after each round."""
    rng = np.random.default_rng(seed)
    d = clients[0][0].shape[1]
    w = np.zeros(d)
    sizes = np.array([len(c[1]) for c in clients], float)
    hist = [global_loss(w, clients)]
    for _ in range(rounds):
        ws = []
        for X, y in clients:
            wl = w.copy()
            for _ in range(local_steps):
                if batch:
                    i = rng.integers(0, len(y), size=batch); wl -= lr * lr_grad(wl, X[i], y[i])
                else:
                    wl -= lr * lr_grad(wl, X, y)
            ws.append(wl)
        w = np.average(ws, axis=0, weights=sizes)
        hist.append(global_loss(w, clients))
    return w, hist


def centralized_gd(clients, steps, lr):
    X = np.vstack([c[0] for c in clients]); y = np.concatenate([c[1] for c in clients])
    w = np.zeros(X.shape[1]); hist = [lr_loss(w, X, y)]
    for _ in range(steps):
        w -= lr * lr_grad(w, X, y); hist.append(lr_loss(w, X, y))
    return w, hist


# ------------------------------------------------------------------- multi-task
class TwoTask:
    """Shared linear feature map W (d -> h), two linear heads; squared loss per task.
    Task 1 has a much larger target scale than task 2 and the targets partly conflict."""

    def __init__(self, d=8, h=4, n=400, seed=0):
        rng = np.random.default_rng(seed)
        self.X = rng.normal(size=(n, d))
        a, b = rng.normal(size=d), rng.normal(size=d)
        self.y1 = 10 * (self.X @ a) + 0.1 * rng.normal(size=n)
        self.y2 = (self.X @ (-a * 0.5 + b)) + 0.1 * rng.normal(size=n)
        self.d, self.h = d, h
        self.rng = rng

    def init(self, seed=1):
        r = np.random.default_rng(seed)
        return [r.normal(size=(self.d, self.h)) * 0.3, r.normal(size=self.h) * 0.3, r.normal(size=self.h) * 0.3]

    def losses(self, P):
        W, v1, v2 = P; Z = self.X @ W
        return np.array([0.5 * np.mean((Z @ v1 - self.y1) ** 2), 0.5 * np.mean((Z @ v2 - self.y2) ** 2)])

    def grads(self, P):
        """Return per-task gradients as flat vectors over ALL parameters (shared + own head)."""
        W, v1, v2 = P; Z = self.X @ W; n = len(self.y1)
        r1, r2 = Z @ v1 - self.y1, Z @ v2 - self.y2
        gW1 = self.X.T @ np.outer(r1, v1) / n; gv1 = Z.T @ r1 / n
        gW2 = self.X.T @ np.outer(r2, v2) / n; gv2 = Z.T @ r2 / n
        g1 = np.concatenate([gW1.ravel(), gv1, np.zeros_like(gv2)])
        g2 = np.concatenate([gW2.ravel(), np.zeros_like(gv1), gv2])
        return g1, g2

    def flat(self, P): return np.concatenate([P[0].ravel(), P[1], P[2]])

    def unflat(self, x):
        k = self.d * self.h
        return [x[:k].reshape(self.d, self.h), x[k:k + self.h], x[k + self.h:]]


def mgda_weights(g1, g2):
    """Min-norm point of the convex hull of {g1, g2}: alpha = clip((g2-g1).g2 / |g1-g2|^2, 0, 1)."""
    diff = g1 - g2
    den = float(diff @ diff)
    a = 0.5 if den == 0 else float(np.clip((g2 - g1) @ g2 / den, 0, 1))
    return a, 1 - a


def train_multitask(prob, method, steps=1500, lr=0.002, seed=1, gn_alpha=1.5, gn_lr=0.025):
    P = prob.init(seed); x = prob.flat(P)
    L0 = prob.losses(P); w = np.array([1.0, 1.0])
    for _ in range(steps):
        P = prob.unflat(x); g1, g2 = prob.grads(P)
        if method == "equal":
            ww = np.array([1.0, 1.0])
        elif method == "mgda":
            a, b = mgda_weights(g1, g2); ww = np.array([a, b]) * 2        # scale so the weights sum to 2 like "equal"
        elif method == "mgda_norm":                                     # MGDA on L2-normalised gradients (Sener & Koltun style)
            n1, n2 = np.linalg.norm(g1) + 1e-12, np.linalg.norm(g2) + 1e-12
            a, b = mgda_weights(g1 / n1, g2 / n2); ww = 2 * np.array([a / n1, b / n2]) * (n1 + n2) / 2
        else:                                                           # GradNorm: pull per-task gradient norms to a target
            L = prob.losses(P); Lt = L / L0
            rate = Lt / Lt.mean()
            G = np.array([np.linalg.norm(w[0] * g1), np.linalg.norm(w[1] * g2)])
            target = G.mean() * rate ** gn_alpha
            gw = np.array([np.sign(G[0] - target[0]) * np.linalg.norm(g1), np.sign(G[1] - target[1]) * np.linalg.norm(g2)])
            w = np.maximum(w - gn_lr * gw, 1e-3); w = 2 * w / w.sum()
            ww = w
        x = x - lr * (ww[0] * g1 + ww[1] * g2)
    return prob.losses(prob.unflat(x))
