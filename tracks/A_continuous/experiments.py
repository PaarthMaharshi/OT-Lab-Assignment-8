"""Track A experiments. Run:  python experiments.py   (about 1-2 minutes)
Writes results/summary.json and results/fig*.png.

Fairness rules used throughout
  * every method gets the same budget of gradient evaluations (samples touched);
  * learning rates are tuned on a grid using TUNING seeds, then re-evaluated on
    different EVALUATION seeds, so the reported number is not a lucky tuning run;
  * stochastic runs report mean +- std over seeds, never the best run.
"""
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optlib import (adagrad, adam, gd, gauss_newton, hvp_fd, momentum, nag,
                    newton, newton_cg, rmsprop)
from problems import (DoubleWell, LogCosh, Logistic, MLP, NonConvexBounded,
                      Quadratic, Rosenbrock, Saddle)

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = {}
LRS = np.logspace(-4, 2, 25)    # wide grid: an earlier 1e-4..10 grid put GD's best lr on the edge
FIRST_ORDER = {"GD": gd, "Momentum": momentum, "NAG": nag, "AdaGrad": adagrad,
               "RMSProp": rmsprop, "Adam": adam}


def safe(v):
    return float(v) if np.isfinite(v) else float("inf")


def last_f(f, xs, tail=1):
    return safe(np.mean([f(x) for x in xs[-tail:]]))


# =========================================================================== 1
def exp1_landscapes():
    dw, sd, q = DoubleWell(), Saddle(), Quadratic(10, 2)
    res = {"double_well": {}, "saddle": {}, "convex": {}}
    for w0 in (-2.0, -0.5, 0.5, 2.0):
        e = gd(dw.grad, [w0], 3000, 0.01)[-1]
        res["double_well"][str(w0)] = {"end": float(e[0]), "f": dw.f(e)}
    for name, x0, kind in [("start (1,0)", [1.0, 0.0], "gd"), ("start (1,1e-3)", [1.0, 1e-3], "gd"),
                           ("start (1,0.5)", [1.0, 0.5], "newton")]:
        xs = gd(sd.grad, x0, 500, 0.1) if kind == "gd" else newton(sd.grad, sd.hess, x0, 30)
        e = xs[-1]
        ev = np.linalg.eigvalsh(sd.hess(e))
        res["saddle"][f"{kind} {name}"] = {"end": e.round(4).tolist(), "f": sd.f(e),
                                           "hessian_eigs": ev.round(3).tolist(),
                                           "type": "saddle" if ev.min() < -1e-6 else "minimum"}
    ends = [gd(q.grad, np.array(s), 800, 0.1)[-1] for s in ([3, 3], [-2, 5], [4, -4])]
    res["convex"] = {"all_reach_same_minimum": bool(np.allclose(ends, 0, atol=1e-6))}
    OUT["exp1_landscapes"] = res

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    w = np.linspace(-2.2, 2.2, 400)
    ax[0].plot(w, [dw.f([v]) for v in w], "k")
    for w0 in (-2.0, -0.5, 0.5, 2.0):
        xs = gd(dw.grad, [w0], 3000, 0.01)[:, 0]
        ax[0].plot(xs[::100], [dw.f([v]) for v in xs[::100]], ".-", ms=4, label=f"start {w0}")
    ax[0].set_title("Non-convex: GD ends in different minima"); ax[0].legend(fontsize=7)
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-2, 2, 200))
    ax[1].contour(X, Y, X ** 2 - Y ** 2 + Y ** 4 / 4, 30, alpha=0.5)
    for name, x0, kind, c in [("GD (1,0): stalls", [1., 0.], "gd", "r"), ("GD (1,1e-3): escapes", [1., 1e-3], "gd", "g"),
                              ("Newton (1,.5): goes to saddle", [1., .5], "newton", "b")]:
        xs = gd(sd.grad, x0, 80, 0.1) if kind == "gd" else newton(sd.grad, sd.hess, x0, 10)
        ax[1].plot(xs[:, 0], xs[:, 1], ".-", c=c, label=name)
    ax[1].set_title("Saddle point"); ax[1].legend(fontsize=7)
    X, Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    ax[2].contour(X, Y, 0.5 * (q.a[0] * X ** 2 + q.a[1] * Y ** 2), 20, alpha=0.5)
    for s in ([3, 3], [-2, 3.5], [3, -3]):
        xs = gd(q.grad, np.array(s, float), 40, 0.1)
        ax[2].plot(xs[:, 0], xs[:, 1], ".-")
    ax[2].set_title("Convex: every start reaches the same minimum")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_landscapes.png"), dpi=110); plt.close(fig)


# =========================================================================== 2
def exp2_convergence_theory():
    out = {}
    q = Quadratic(10, 8); x0 = np.ones(8)
    xs = gd(q.grad, x0, 80, 1 / q.L)
    fk = np.array([q.f(x) for x in xs]); k = np.arange(81)
    b1 = (1 - q.mu / q.L) ** (2 * k) * q.f(x0)
    out["strongly_convex"] = {"bound": "f_k <= (1-mu/L)^(2k) f_0", "max_ratio_f_over_bound": float((fk / b1).max())}
    p = LogCosh(); x0c = np.array([3.0, -2.0, 1.5])
    xc = gd(p.grad, x0c, 400, 1 / p.L)
    fc = np.array([p.f(x) for x in xc]); kk = np.arange(1, 401)
    b2 = p.L * (x0c @ x0c) / (2 * kk)
    out["convex"] = {"bound": "f_k - f* <= L|x0-x*|^2/(2k)", "max_ratio_f_over_bound": float((fc[1:] / b2).max())}
    n = NonConvexBounded(); x0n = np.array([0.3, -1.2, 2.0]); K = 400
    xn = gd(n.grad, x0n, K, 1 / n.L)
    gn = np.array([np.linalg.norm(n.grad(x)) for x in xn[:K]])
    run_min = np.minimum.accumulate(gn)
    b3 = np.sqrt(2 * n.L * n.f(x0n) / np.arange(1, K + 1))
    out["nonconvex"] = {"bound": "min_k |grad| <= sqrt(2L(f0-f*)/K)", "max_ratio": float((run_min / b3).max()),
                        "note": "only a stationary point is guaranteed, not a minimum"}
    OUT["exp2_convergence_theory"] = out
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].semilogy(k, fk + 1e-300, label="GD"); ax[0].semilogy(k, b1, "--", label="bound"); ax[0].set_title("Strongly convex: linear rate")
    ax[1].loglog(kk, fc[1:], label="GD"); ax[1].loglog(kk, b2, "--", label="L|x0|^2/2k"); ax[1].set_title("Convex: O(1/k)")
    ax[2].loglog(np.arange(1, K + 1), run_min, label="min |grad|"); ax[2].loglog(np.arange(1, K + 1), b3, "--", label="bound"); ax[2].set_title("Non-convex: gradient norm")
    for a in ax: a.legend(); a.set_xlabel("iteration k")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_convergence_theory.png"), dpi=110); plt.close(fig)


# =========================================================================== 3
def tune(run, f, lrs, seeds):
    """run(lr, seed) -> iterates. Returns best lr and the lr -> mean final f curve."""
    curve = []
    for lr in lrs:
        curve.append(np.mean([last_f(f, run(lr, s)) for s in seeds]))
    curve = np.array(curve)
    return float(lrs[int(np.nanargmin(np.where(np.isfinite(curve), curve, np.inf)))]), curve


def steps_to(f, xs, target):
    for i, x in enumerate(xs):
        if f(x) <= target:
            return i
    return None


def exp3_shootout():
    out = {}
    curves = {}
    # ---- (a) ill-conditioned quadratic, kappa = 100, 300 steps, full batch
    q = Quadratic(100, 20)
    x0 = np.random.default_rng(0).normal(size=20)   # NOT all-ones: AdaGrad's first step lr*g/|g| then lands exactly on 0
    rows = {}
    for name, opt in FIRST_ORDER.items():
        lr, curve = tune(lambda lr, s: opt(q.grad, x0, 300, lr), q.f, LRS, [0])
        t = time.perf_counter(); xs = opt(q.grad, x0, 300, lr); dt = time.perf_counter() - t
        rows[name] = {"lr": lr, "final_subopt": last_f(q.f, xs), "steps_to_1e-6": steps_to(q.f, xs, 1e-6), "seconds": dt}
        curves[("quadratic", name)] = (curve, xs)
    t = time.perf_counter(); xs = newton(q.grad, q.hess, x0, 5); dt = time.perf_counter() - t
    rows["Newton"] = {"lr": 1.0, "final_subopt": last_f(q.f, xs), "steps_to_1e-6": steps_to(q.f, xs, 1e-6), "seconds": dt}
    out["quadratic_kappa100"] = rows
    # ---- (b) Rosenbrock (non-convex), 5000 steps
    r = Rosenbrock(); x0r = np.array([-1.2, 1.0]); rows = {}
    for name, opt in FIRST_ORDER.items():
        lr, curve = tune(lambda lr, s: opt(r.grad, x0r, 5000, lr), r.f, LRS, [0])
        xs = opt(r.grad, x0r, 5000, lr)
        rows[name] = {"lr": lr, "final_subopt": last_f(r.f, xs), "steps_to_1e-6": steps_to(r.f, xs, 1e-6)}
    xs = newton(r.grad, r.hess, x0r, 50)
    rows["Newton"] = {"lr": 1.0, "final_subopt": last_f(r.f, xs), "steps_to_1e-6": steps_to(r.f, xs, 1e-6)}
    out["rosenbrock_nonconvex"] = rows
    # ---- (c) logistic regression, equal budget of 30 epochs, stochastic methods use bs=32
    lg = Logistic(n=2000, d=20, lam=1e-3, seed=0); bs = 32; epochs = 30
    sub = lambda w: lg.f(w) - lg.fstar
    rows = {}
    full_steps, mb_steps = epochs, epochs * lg.n // bs
    tune_seeds, eval_seeds = [0, 1, 2], [10, 11, 12, 13, 14]

    def mk(opt, steps, batch):
        def run(lr, seed):
            lg.rng = np.random.default_rng(seed)
            g = lg.grad if batch is None else lg.batch_grad(batch)
            return opt(g, np.zeros(20), steps, lr)
        return run
    for name, opt in FIRST_ORDER.items():
        for tag, steps, batch in (("full-batch", full_steps, None), (f"mini-batch {bs}", mb_steps, bs)):
            run = mk(opt, steps, batch)
            seeds_t, seeds_e = ([0], [0]) if batch is None else (tune_seeds, eval_seeds)
            lr, curve = tune(run, sub, LRS, seeds_t)
            vals, secs = [], []
            for s in seeds_e:
                t = time.perf_counter(); xs = run(lr, s); secs.append(time.perf_counter() - t)
                vals.append(last_f(sub, xs, tail=max(1, steps // 20)))
            rows[f"{name} ({tag})"] = {"lr": lr, "final_subopt_mean": float(np.mean(vals)), "final_subopt_std": float(np.std(vals)),
                                       "seconds": float(np.mean(secs))}
            curves[("logistic", f"{name} ({tag})")] = (curve, None)
    t = time.perf_counter(); xs = newton(lg.grad, lg.hess, np.zeros(20), 8); dt = time.perf_counter() - t
    rows["Newton (full-batch, 8 steps)"] = {"lr": 1.0, "final_subopt_mean": last_f(sub, xs), "final_subopt_std": 0.0, "seconds": dt}
    out["logistic_regression_30_epochs"] = rows
    OUT["exp3_shootout"] = out

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in FIRST_ORDER:
        c, xs = curves[("quadratic", name)]
        ax[0].loglog(LRS, np.where(np.isfinite(c), c, np.nan) + 1e-20, label=name)
        ax[1].semilogy([q.f(x) + 1e-20 for x in xs], label=name)
    ax[0].set_ylim(1e-20, 1e4); ax[0].set_title("Quadratic k=100: final f vs learning rate (diverged runs clipped)"); ax[0].set_xlabel("lr"); ax[0].legend(fontsize=7)
    ax[1].set_title("Quadratic k=100: f(x_k) at tuned lr"); ax[1].set_xlabel("step"); ax[1].legend(fontsize=7)
    for name in ("GD", "Momentum", "Adam", "RMSProp"):
        c, _ = curves[("logistic", f"{name} (mini-batch {bs})")]
        ax[2].loglog(LRS, np.where(np.isfinite(c), c, np.nan), label=f"{name} (mb {bs})")
    ax[2].set_title("Logistic: mini-batch sensitivity to lr"); ax[2].set_xlabel("lr"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_shootout.png"), dpi=110); plt.close(fig)


# =========================================================================== 4
def exp4_backprop_mlp():
    out = {}
    m = MLP(h=16, lam=0.0, seed=0)
    th = np.random.default_rng(5).normal(size=m.dim) * 0.3
    g = m.grad(th)
    gfd = np.array([(m.loss(th + e) - m.loss(th - e)) / 2e-6 for e in np.eye(m.dim) * 1e-6])
    out["gradient_check_max_rel_err"] = float(np.abs(g - gfd).max() / np.abs(gfd).max())
    epochs, bs = 100, 32
    res, curves = {}, {}
    for name, opt, batch in (("GD full-batch", gd, None), ("SGD bs32", gd, bs), ("Adam bs32", adam, bs)):
        steps = epochs if batch is None else epochs * 400 // bs
        def run(lr, seed, opt=opt, batch=batch, steps=steps):
            mm = MLP(h=16, seed=seed)
            gr = mm.grad if batch is None else mm.batch_grad(batch)
            return opt(gr, mm.theta0, steps, lr)
        best, bestv = None, np.inf
        for l in LRS:
            v = np.mean([safe(MLP(h=16, seed=s).loss(run(l, s)[-1])) for s in (0, 1, 2)])
            if v < bestv: best, bestv = float(l), v
        accs, losses = [], []
        for s in (10, 11, 12, 13, 14):
            xs = run(best, s); mm = MLP(h=16, seed=s)
            accs.append(mm.accuracy(xs[-1])); losses.append(mm.loss(xs[-1]))
            if s == 10: curves[name] = [mm.loss(x) for x in xs[::max(1, len(xs) // 100)]]
        res[name] = {"lr": best, "train_loss_mean": float(np.mean(losses)), "train_loss_std": float(np.std(losses)),
                     "test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs))}
    out["training_100_epochs"] = res
    reg = {}
    for lam in (0.0, 1e-3, 1e-2, 1e-1):
        accs, norms, losses = [], [], []
        for s in (10, 11, 12):
            mm = MLP(h=16, lam=lam, seed=s)
            th = adam(mm.batch_grad(bs), mm.theta0, 100 * 400 // bs, res["Adam bs32"]["lr"])[-1]
            W1, _, W2, _ = mm.unpack(th)
            accs.append(mm.accuracy(th)); norms.append(float(np.linalg.norm(W1) + np.linalg.norm(W2)))
            losses.append(MLP(h=16, lam=0.0, seed=s).loss(th))
        reg[str(lam)] = {"test_acc": float(np.mean(accs)), "weight_norm": float(np.mean(norms)), "data_loss": float(np.mean(losses))}
    out["l2_regularisation_adam"] = reg
    OUT["exp4_backprop_mlp"] = out
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in curves.items():
        ax.semilogy(np.linspace(0, 100, len(c)), c, label=name)
    ax.set_xlabel("% of budget (100 epochs)"); ax.set_ylabel("train loss"); ax.legend(); ax.set_title("MLP (backprop gradients)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_mlp.png"), dpi=110); plt.close(fig)


# =========================================================================== 5
def exp5_noise():
    lg = Logistic(n=2000, d=20, lam=1e-3, seed=0)
    sub = lambda w: lg.f(w) - lg.fstar
    epochs, eta0 = 20, 0.5 / lg.L_global * 2
    out, rows = {"eta0": eta0, "epochs": epochs}, {}
    for bs in (1, 8, 64, 2000):
        steps = epochs * lg.n // bs
        for sched in ("constant", "decay"):
            vals = []
            for seed in range(5):
                lg.rng = np.random.default_rng(seed)
                g = lg.batch_grad(bs)
                if sched == "constant":
                    xs = gd(g, np.zeros(20), steps, eta0)
                else:
                    tau = max(1, steps // 10)
                    x = np.zeros(20); xs = [x]
                    for t in range(steps):
                        x = x - eta0 / (1 + t / tau) * g(x); xs.append(x)
                vals.append(np.mean([sub(x) for x in xs[-max(1, steps // 10):]]))
            rows[f"bs={bs} {sched}"] = {"noise_floor_mean": float(np.mean(vals)), "std": float(np.std(vals))}
    out["rows"] = rows
    OUT["exp5_noisy_objective"] = out
    fig, ax = plt.subplots(figsize=(6, 4))
    for sched, mk in (("constant", "o-"), ("decay", "s--")):
        ax.loglog([1, 8, 64, 2000], [rows[f"bs={b} {sched}"]["noise_floor_mean"] for b in (1, 8, 64, 2000)], mk, label=sched)
    ax.set_xlabel("batch size"); ax.set_ylabel("f - f* (last 10% of steps)"); ax.legend(); ax.set_title("Noise floor of SGD, equal 20-epoch budget")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_noise.png"), dpi=110); plt.close(fig)


# =========================================================================== 6
def exp6_curvature():
    lg = Logistic(n=2000, d=20, lam=1e-3, seed=0)
    sub = lambda w: lg.f(w) - lg.fstar
    out = {}
    w = np.random.default_rng(2).normal(size=20) * 0.3
    v = np.random.default_rng(3).normal(size=20)
    H = lg.hess(w)
    out["hvp"] = {"exact_matrix_free_err": float(np.linalg.norm(lg.hvp(w, v) - H @ v) / np.linalg.norm(H @ v)),
                  "finite_difference_err": float(np.linalg.norm(hvp_fd(lg.grad, w, v) - H @ v) / np.linalg.norm(H @ v)),
                  "cost": "2 gradient calls, O(nd), no d x d matrix formed"}
    Hs = lg.hess(lg.wstar)
    ev, U = np.linalg.eigh(Hs)
    order = np.argsort(ev)[::-1]; ev, U = ev[order], U[:, order]
    lowrank = {}
    for r in (1, 2, 3, 5, 10):
        Hr = (U[:, :r] * ev[:r]) @ U[:, :r].T
        lowrank[str(r)] = float(np.linalg.norm(Hs - Hr) / np.linalg.norm(Hs))
    from scipy.sparse.linalg import LinearOperator, eigsh
    op = LinearOperator((20, 20), matvec=lambda u: lg.hvp(lg.wstar, u), dtype=float)
    top = np.sort(eigsh(op, k=5, which="LA", return_eigenvectors=False))[::-1]
    out["low_rank_hessian"] = {"rel_frobenius_error_by_rank": lowrank, "condition_number": float(ev[0] / ev[-1]),
                               "top5_eig_via_HVP_only_matches_eigh": bool(np.allclose(top, ev[:5], rtol=1e-6))}
    F, Fe = lg.fisher_true(lg.wstar), lg.fisher_empirical(lg.wstar)
    wf = 3 * lg.wstar
    rel = lambda A, B: float(np.linalg.norm(A - B) / np.linalg.norm(B))
    out["fisher"] = {"true_fisher_vs_hessian_at_optimum": rel(F + lg.lam * np.eye(20), Hs),
                     "empirical_fisher_vs_hessian_at_optimum": rel(Fe + lg.lam * np.eye(20), Hs),
                     "empirical_fisher_vs_hessian_far_from_optimum": rel(lg.fisher_empirical(wf) + lg.lam * np.eye(20), lg.hess(wf))}
    tol = 1e-10; cap = 5000; w0 = np.zeros(20)
    def reach(xs): return steps_to(sub, xs, tol)
    rows = {}
    xs = newton(lg.grad, lg.hess, w0, 20); rows["Newton (exact Hessian)"] = {"steps": reach(xs), "cost_note": "d x d solve each step"}
    xs, calls = newton_cg(lg.grad, lg.hvp, w0, 20); rows["Newton-CG (HVP only)"] = {"steps": reach(xs), "hvp_calls": calls}
    def natgrad(x):
        return x - np.linalg.solve(lg.fisher_true(x) + lg.lam * np.eye(20), lg.grad(x))
    x = w0.copy(); xsn = [x]
    for _ in range(20): x = natgrad(x); xsn.append(x)
    rows["Natural gradient (true Fisher + lam I)"] = {"steps": reach(xsn),
                                                      "same_iterates_as_Newton": bool(np.allclose(np.array(xsn), newton(lg.grad, lg.hess, w0, 20)))}
    def dfisher(x):
        return x - 1.0 * lg.grad(x) / (np.diag(lg.fisher_true(x)) + lg.lam + 1e-1)
    x = w0.copy(); xsd = [x]
    for _ in range(cap): x = dfisher(x); xsd.append(x)
    rows["Diagonal-Fisher preconditioned GD"] = {"steps": reach(xsd), "final_subopt": safe(sub(xsd[-1]))}
    lr, _ = tune(lambda l, s: gd(lg.grad, w0, cap, l), sub, LRS, [0])
    xs = gd(lg.grad, w0, cap, lr); rows["GD (tuned lr)"] = {"steps": reach(xs), "lr": lr, "final_subopt": safe(sub(xs[-1]))}
    lr, _ = tune(lambda l, s: adam(lg.grad, w0, cap, l), sub, LRS, [0])
    xs = adam(lg.grad, w0, cap, lr); rows["Adam (tuned lr)"] = {"steps": reach(xs), "lr": lr, "final_subopt": safe(sub(xs[-1]))}
    out["steps_to_f-f*<1e-10 (None = not reached in 5000)"] = rows
    OUT["exp6_curvature"] = out


# =========================================================================== 7
def exp7_gauss_newton():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 40); a, b = 2.0, -0.8
    y = a * np.exp(b * t) + 0.05 * rng.normal(size=40)
    res = lambda x: x[0] * np.exp(x[1] * t) - y
    jac = lambda x: np.stack([np.exp(x[1] * t), x[0] * t * np.exp(x[1] * t)], axis=1)
    cost = lambda x: 0.5 * float(res(x) @ res(x))
    grad = lambda x: jac(x).T @ res(x)
    def hess(x):
        r = res(x); e = np.exp(x[1] * t)
        S = np.array([[0.0, np.sum(r * t * e)], [np.sum(r * t * e), np.sum(r * x[0] * t ** 2 * e)]])
        return jac(x).T @ jac(x) + S
    x0 = np.array([1.0, -0.2])
    xopt = gauss_newton(res, jac, x0, 100, lm=1e-6)[-1]
    fopt = cost(xopt)
    rows = {}
    def summarize(name, xs):
        gn = [np.linalg.norm(grad(x)) for x in xs]
        k = next((i for i, g in enumerate(gn) if g < 1e-8), None)
        rows[name] = {"iterations_to_grad<1e-8": k, "final_cost_minus_best": safe(cost(xs[-1]) - fopt)}
    summarize("Gauss-Newton", gauss_newton(res, jac, x0, 50))
    summarize("Levenberg-Marquardt (lm=1e-2)", gauss_newton(res, jac, x0, 50, lm=1e-2))
    summarize("Newton (full Hessian)", newton(grad, hess, x0, 50))
    lr, _ = tune(lambda l, s: gd(grad, x0, 2000, l), cost, LRS, [0])
    summarize(f"GD 2000 steps (lr={lr:.3g})", gd(grad, x0, 2000, lr))
    OUT["exp7_gauss_newton"] = {"problem": "fit y=a*exp(b t), truth (2,-0.8), noise 0.05", "fitted": xopt.round(4).tolist(), "rows": rows}


if __name__ == "__main__":
    for fn in (exp1_landscapes, exp2_convergence_theory, exp3_shootout, exp4_backprop_mlp,
               exp5_noise, exp6_curvature, exp7_gauss_newton):
        t = time.perf_counter(); fn(); print(f"{fn.__name__}: {time.perf_counter() - t:.1f}s")
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print(json.dumps(OUT, indent=1))
