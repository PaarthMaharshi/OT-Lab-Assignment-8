"""Track B experiments. Run: python experiments.py  -> results/summary.json, results/fig*.png,
results/maxcut_instances.json (reused by Track C).

Fairness: exact methods must agree on the optimum (asserted); heuristics are scored against
the proven optimum when one exists, otherwise against the best value any method found;
stochastic methods use repeated seeds and report mean +- std."""
import json
import os
import statistics as st
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import featsel
import knapsack as kn
import qubo as qb
import scheduling as sc

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = {}


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t


# ================================================================ knapsack
def exp_knapsack():
    rows = {}
    for kind in ("uncorrelated", "strongly_correlated"):
        for n in (20, 40, 80, 160):
            acc = {"greedy_s": [], "dp_s": [], "bnb_s": [], "ilp_s": [], "bnb_nodes": [], "bnb_proven": [], "greedy_gap_pct": []}
            for seed in range(3):
                v, w, cap = kn.make_instance(n, kind, seed)
                d, t_dp = timed(kn.dp, v, w, cap)
                (b, nodes, proven), t_bnb = timed(kn.branch_and_bound, v, w, cap, 150_000)
                (i, ilp_ok), t_ilp = timed(kn.ilp, v, w, cap, 30)
                g, t_g = timed(kn.greedy_ratio, v, w, cap)
                opt = kn.value_of(v, d)
                assert kn.feasible(w, d, cap) and kn.feasible(w, b, cap) and kn.feasible(w, i, cap) and kn.feasible(w, g, cap)
                if ilp_ok:
                    assert kn.value_of(v, i) == opt, "ILP and DP disagree"
                if proven:
                    assert kn.value_of(v, b) == opt, "B&B and DP disagree"
                acc["greedy_s"].append(t_g); acc["dp_s"].append(t_dp); acc["bnb_s"].append(t_bnb); acc["ilp_s"].append(t_ilp)
                acc["bnb_nodes"].append(nodes); acc["bnb_proven"].append(proven)
                acc["greedy_gap_pct"].append(100 * (opt - kn.value_of(v, g)) / opt)
            rows[f"{kind} n={n}"] = {"greedy_ms": 1e3 * st.mean(acc["greedy_s"]), "dp_ms": 1e3 * st.mean(acc["dp_s"]),
                                     "bnb_ms": 1e3 * st.mean(acc["bnb_s"]), "ilp_ms": 1e3 * st.mean(acc["ilp_s"]),
                                     "bnb_nodes_mean": st.mean(acc["bnb_nodes"]), "bnb_proven": f"{sum(acc['bnb_proven'])}/3",
                                     "greedy_gap_pct_mean": st.mean(acc["greedy_gap_pct"]), "greedy_gap_pct_max": max(acc["greedy_gap_pct"])}
    OUT["knapsack"] = rows
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for kind, ls in (("uncorrelated", "-"), ("strongly_correlated", "--")):
        ns = [20, 40, 80, 160]
        for m, c in (("greedy_ms", "C0"), ("dp_ms", "C1"), ("bnb_ms", "C2"), ("ilp_ms", "C3")):
            ax[0].semilogy(ns, [rows[f"{kind} n={n}"][m] for n in ns], ls, color=c, label=f"{m[:-3]} ({kind[:5]})")
        ax[1].semilogy(ns, [rows[f"{kind} n={n}"]["bnb_nodes_mean"] for n in ns], ls, label=kind)
        ax[2].plot(ns, [rows[f"{kind} n={n}"]["greedy_gap_pct_mean"] for n in ns], ls, label=kind)
    ax[0].set_title("Runtime (ms)"); ax[0].legend(fontsize=6); ax[1].set_title("B&B nodes (cap 150k)"); ax[1].legend()
    ax[2].set_title("Greedy gap to optimum (%)"); ax[2].legend()
    for a in ax: a.set_xlabel("items n")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_knapsack.png"), dpi=110); plt.close(fig)


# ================================================================ Max-Cut
def exp_maxcut():
    rows, ratios = {}, {}
    for n, graphs, sweeps in ((10, 3, 300), (14, 3, 300), (18, 3, 400), (22, 3, 400), (30, 3, 500), (50, 3, 500), (100, 3, 1000)):
        agg = {"ilp_s": [], "ilp_proven": [], "sa": [], "sa_s": [], "ls": [], "ls_s": [], "bf_s": []}
        for g in range(graphs):
            edges = qb.random_graph(n, 0.5, seed=100 * n + g)
            Q = qb.maxcut_qubo(n, edges)
            (cut_ilp, proven, _), t_ilp = timed(qb.ilp_maxcut, n, edges, 20)
            ref, ref_kind = cut_ilp, "ILP"
            if n <= 22:
                (e, _), t_bf = timed(qb.brute_force_qubo, Q)
                agg["bf_s"].append(t_bf)
                assert proven and cut_ilp == -round(e), "ILP and brute force disagree"
                ref, ref_kind = -round(e), "brute force (exact)"
            sa_runs, ls_runs = [], []
            t = time.perf_counter()
            for s in range(10):
                sa_runs.append(-qb.sa_qubo(Q, sweeps=sweeps, T0=2.0, T1=0.05, seed=s)[0])
            sa_t = (time.perf_counter() - t) / 10
            t = time.perf_counter()
            for s in range(10):
                ls_runs.append(-qb.local_search_qubo(Q, restarts=20, seed=s)[0])
            ls_t = (time.perf_counter() - t) / 10
            ref = max(ref, max(sa_runs), max(ls_runs))          # best known (ILP may time out at n=100)
            agg["ilp_s"].append(t_ilp); agg["ilp_proven"].append(proven)
            agg["sa"] += [c / ref for c in sa_runs]; agg["ls"] += [c / ref for c in ls_runs]
            agg["sa_s"].append(sa_t); agg["ls_s"].append(ls_t)
            agg.setdefault("ilp_ratio", []).append(cut_ilp / ref)
        rows[f"n={n}"] = {"ILP_s": st.mean(agg["ilp_s"]), "ILP_proven": f"{sum(agg['ilp_proven'])}/{graphs}",
                          "ILP_cut/best": st.mean(agg["ilp_ratio"]),
                          "SA_cut/best_mean": st.mean(agg["sa"]), "SA_cut/best_min": min(agg["sa"]), "SA_s": st.mean(agg["sa_s"]),
                          "LS_cut/best_mean": st.mean(agg["ls"]), "LS_cut/best_min": min(agg["ls"]), "LS_s": st.mean(agg["ls_s"]),
                          "bruteforce_s": st.mean(agg["bf_s"]) if agg["bf_s"] else None}
        ratios[n] = (st.mean(agg["sa"]), st.mean(agg["ls"]))
    OUT["maxcut"] = rows
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ns = list(ratios)
    ax[0].plot(ns, [ratios[n][0] for n in ns], "o-", label="SA"); ax[0].plot(ns, [ratios[n][1] for n in ns], "s--", label="1-flip LS")
    ax[0].set_ylabel("cut / best known"); ax[0].set_xlabel("nodes"); ax[0].legend(); ax[0].set_title("Max-Cut quality (G(n,0.5), 30 runs)")
    ax[1].semilogy(ns, [rows[f"n={n}"]["ILP_s"] for n in ns], "o-", label="ILP")
    ax[1].semilogy(ns, [rows[f"n={n}"]["SA_s"] for n in ns], "s-", label="SA")
    ax[1].semilogy(ns, [rows[f"n={n}"]["LS_s"] for n in ns], "^-", label="LS")
    bf = [(n, rows[f"n={n}"]["bruteforce_s"]) for n in ns if rows[f"n={n}"]["bruteforce_s"]]
    ax[1].semilogy(*zip(*bf), "x-", label="brute force"); ax[1].legend(); ax[1].set_xlabel("nodes"); ax[1].set_title("Time per instance (s)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_maxcut.png"), dpi=110); plt.close(fig)
    # small instances saved for Track C (QAOA)
    inst = []
    for n in (6, 8, 10, 12):
        for g in range(3):
            edges = qb.random_graph(n, 0.5, seed=7000 + 10 * n + g)
            e, x = qb.brute_force_qubo(qb.maxcut_qubo(n, edges))
            inst.append({"n": n, "seed": 7000 + 10 * n + g, "edges": edges, "max_cut": -round(e), "optimal_x": x.astype(int).tolist()})
    with open(os.path.join(RES, "maxcut_instances.json"), "w") as f:
        json.dump(inst, f)


# ================================================================ feature selection
def split(seed):
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
    sc_ = StandardScaler().fit(Xtr)
    return sc_.transform(Xtr), sc_.transform(Xva), sc_.transform(Xte), ytr, yva, yte


def exp_featsel():
    K, D = 4, 30
    rows = {m: {"val": [], "test": [], "fits": [], "s": []} for m in ("forward", "backward", "lasso(L1)", "QUBO+SA", "random")}
    exh = {}
    repaired = [0]
    for seed in range(5):
        Xtr, Xva, Xte, ytr, yva, yte = split(seed)
        mk = lambda S, X=Xtr: LogisticRegression(max_iter=200).fit(Xtr[:, list(S)], ytr)
        val = lambda S: mk(S).score(Xva[:, list(S)], yva)
        test = lambda S: mk(S).score(Xte[:, list(S)], yte)

        def record(name, S, fits, secs):
            rows[name]["val"].append(val(S)); rows[name]["test"].append(test(S)); rows[name]["fits"].append(fits); rows[name]["s"].append(secs)
        c = featsel.Counted(val); S, t = timed(featsel.greedy_forward, c, D, K); record("forward", S, c.calls, t)
        c = featsel.Counted(val); S, t = timed(featsel.backward_elimination, c, D, K); record("backward", S, c.calls, t)

        lasso_fits = [0]

        def lasso():
            for C in np.logspace(-3, 1, 80):
                lasso_fits[0] += 1
                m = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xtr, ytr)
                if (m.coef_ != 0).sum() >= K:
                    return tuple(sorted(np.argsort(-np.abs(m.coef_[0]))[:K]))
        S, t = timed(lasso); record("lasso(L1)", S, lasso_fits[0], t)
        Q, const, r, Cm = featsel.feature_qubo(Xtr, ytr.astype(float), K, alpha=0.5, P=0.3)   # P from the sweep (training-data objective only)
        def qubo_sa():
            best = min((qb.sa_qubo(Q, sweeps=300, T0=1.0, T1=0.01, seed=s) for s in range(10)), key=lambda z: z[0])
            return tuple(np.where(best[1] > 0.5)[0])
        S, t = timed(qubo_sa)
        if len(S) != K:                                              # infeasible QUBO answer: repair by relevance
            repaired[0] += 1
            S = tuple(sorted(sorted(S, key=lambda j: -r[j])[:K])) if len(S) > K else tuple(sorted(set(S) | set(np.argsort(-r)[:K - len(S)])))
        record("QUBO+SA", S, 0, t)
        rng = np.random.default_rng(seed)
        rv, rt = [], []
        for _ in range(50):
            S = tuple(sorted(rng.choice(D, K, replace=False))); rv.append(val(S)); rt.append(test(S))
        rows["random"]["val"].append(st.mean(rv)); rows["random"]["test"].append(st.mean(rt)); rows["random"]["fits"].append(0); rows["random"]["s"].append(0)
        if seed == 0:
            c = featsel.Counted(val)
            S, t = timed(featsel.exhaustive, c, D, K)
            exh = {"subset": list(map(int, S)), "val": val(S), "test": test(S), "fits": c.calls, "seconds": t}
    summ = {m: {"val_mean": st.mean(r["val"]), "test_mean": st.mean(r["test"]), "test_std": st.pstdev(r["test"]),
                "model_fits": st.mean(r["fits"]), "seconds": st.mean(r["s"])} for m, r in rows.items()}
    OUT["feature_selection"] = {"data": "sklearn breast cancer, d=30, k=4, 5 random splits (60/20/20), logistic regression",
                                "note": "test set has ~114 samples: +-2 points is noise", "methods": summ,
                                "exhaustive_split0_only": exh, "qubo_answers_needing_repair_of_5": repaired[0], "formula_fits_forward_d30_k4": featsel.n_fits_forward(30, 4)}


def exp_qubo_penalty():
    Xtr, Xva, Xte, ytr, yva, yte = split(0)
    D, K, alpha = 30, 5, 0.5
    r = np.abs([np.corrcoef(Xtr[:, i], ytr)[0, 1] for i in range(D)])
    Cm = np.abs(np.corrcoef(Xtr, rowvar=False))
    import itertools
    combos = np.array(list(itertools.combinations(range(D), K)))
    obj = -r[combos].sum(1) + alpha * sum(Cm[combos[:, a], combos[:, b]] for a in range(K) for b in range(a + 1, K))
    best_feasible = float(obj.min())
    rows = {}
    for P in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        Q, const, _, _ = featsel.feature_qubo(Xtr, ytr.astype(float), K, alpha=alpha, P=P)
        sizes, gaps = [], []
        for s in range(20):
            e, x = qb.sa_qubo(Q, sweeps=300, T0=1.0, T1=0.01, seed=s)
            S = np.where(x > 0.5)[0]; sizes.append(len(S))
            if len(S) == K:
                gaps.append(float((-r[S].sum() + alpha * sum(Cm[a, b] for a, b in itertools.combinations(S, 2))) - best_feasible))
        rows[f"P={P}"] = {"runs_with_exactly_k_features": f"{sum(s == K for s in sizes)}/20", "mean_size": st.mean(sizes),
                          "mean_gap_to_exact_feasible_optimum": st.mean(gaps) if gaps else None}
    OUT["qubo_penalty_sweep"] = {"k": K, "alpha": alpha, "exact_best_feasible_objective": best_feasible, "rows": rows,
                                 "note": "P too small -> constraint ignored; P large enough -> always feasible"}


# ================================================================ scheduling (DP)
def exp_intervals():
    rows = {}
    for n in (10, 200, 2000):
        res = {"dp": [], "earliest_finish": [], "max_weight": []}
        for seed in range(5):
            jobs = sc.make_jobs(n, seed, horizon=100 * n // 10 + 100)
            opt = sc.weighted_dp(jobs)
            if n <= 200:
                assert sc.ilp(jobs) == opt
            res["dp"].append(opt)
            res["earliest_finish"].append(sc.greedy_earliest_finish(jobs) / opt)
            res["max_weight"].append(sc.greedy_max_weight(jobs) / opt)
        rows[f"n={n}"] = {"greedy_earliest_finish/opt": st.mean(res["earliest_finish"]), "greedy_max_weight/opt": st.mean(res["max_weight"]),
                          "ILP_checked": n <= 200}
    OUT["weighted_interval_scheduling"] = rows


if __name__ == "__main__":
    import sys
    allf = {f.__name__: f for f in (exp_knapsack, exp_maxcut, exp_featsel, exp_qubo_penalty, exp_intervals)}
    path = os.path.join(RES, "summary.json")
    if len(sys.argv) > 1 and os.path.exists(path):             # rerun only the named experiments, keep the rest
        OUT.update(json.load(open(path)))
    for name in (sys.argv[1:] or allf):
        t = time.perf_counter(); allf[name](); print(f"{name}: {time.perf_counter() - t:.1f}s", flush=True)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(OUT, f, indent=1)
