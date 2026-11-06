"""Track C experiments (ALL SIMULATED on a numpy simulator; no real quantum hardware was used).
Run: python experiments.py [exp_name ...]  -> results/summary.json, results/fig*.png"""
import json
import os
import statistics as st
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import qopt
import qsim

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B_RES = os.path.join(HERE, "..", "B_discrete", "results", "maxcut_instances.json")
os.makedirs(RES, exist_ok=True)
OUT = {}
EDGES4 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]


# ================================================================== VQE
def exp_vqe():
    n, g = 4, 1.0
    H = qopt.tfim_dense(n, g)
    e0 = float(np.linalg.eigvalsh(H)[0])
    out = {"hamiltonian": f"open TFIM chain n={n}, H = -sum ZZ - {g} sum X", "exact_ground_energy": e0}
    # (a) expressivity vs depth (exact expectation, L-BFGS, 6 restarts)
    rows = {}
    for L in (1, 2, 3, 4):
        ops = qopt.hea_ops(n, L); P = qopt.n_params(ops)
        f = lambda t: qsim.expval_dense(qsim.run(ops, t, n), H)
        rng = np.random.default_rng(L)
        best = min(minimize(f, rng.uniform(0, 2 * np.pi, P), method="BFGS").fun for _ in range(6))
        rows[f"layers={L}"] = {"params": P, "energy_error": best - e0}
    out["error_vs_layers"] = rows
    # (b) optimizer comparison at L=3 (P=16). Hyperparameters tuned on TUNING seeds, reported on different EVAL seeds,
    #     compared at an equal budget of circuit executions.
    L = 3; ops = qopt.hea_ops(n, L); P = qopt.n_params(ops)
    f = lambda t: qsim.expval_dense(qsim.run(ops, t, n), H)
    grad = lambda t: qsim.param_shift_grad(ops, t, n, lambda s_: qsim.expval_dense(s_, H))
    metric = lambda t: qsim.metric_tensor(ops, t, n)
    BUDGET = 4800                                           # = 150 parameter-shift gradients (2 * 16 executions each)

    def err_at(h, budget):
        ok = [x for x in h if x[0] <= budget]
        return (ok[-1][1] if ok else h[0][1]) - e0

    def run_opt(name, hp, th0, seed):
        if name == "GD+param-shift": return qopt.gd_ps(f, grad, th0, 150, **hp)[1]
        if name == "Adam+param-shift": return qopt.adam_ps(f, grad, th0, 150, **hp)[1]
        if name == "SPSA": return qopt.spsa(f, th0, 2400, a=hp["a"], c=0.15, seed=seed)[1]
        return qopt.qng(f, grad, metric, th0, 40, lr=hp["lr"], reg=hp["reg"], metric_cost=2 * P * P)[1]
    lrs = (0.02, 0.05, 0.1, 0.2, 0.4)
    grids = {"GD+param-shift": [{"lr": l} for l in lrs], "Adam+param-shift": [{"lr": l} for l in lrs],
             "SPSA": [{"a": a_} for a_ in (0.05, 0.1, 0.3, 0.6, 1.0)],
             "QNG": [{"lr": l, "reg": r} for l in (0.05, 0.1, 0.2, 0.5) for r in (1e-3, 1e-2, 1e-1)]}
    runs, curves = {}, {}
    for name, grid in grids.items():
        scores = []
        for hp in grid:
            scores.append(np.mean([err_at(run_opt(name, hp, np.random.default_rng(200 + sd).uniform(0, 2 * np.pi, P), sd), BUDGET) for sd in range(3)]))
        hp = grid[int(np.argmin(scores))]
        errs, circ, its = [], [], []
        for sd in range(5):
            h = run_opt(name, hp, np.random.default_rng(100 + sd).uniform(0, 2 * np.pi, P), sd)
            e = np.array([x[1] for x in h]) - e0; c = np.array([x[0] for x in h])
            errs.append(float(err_at(h, BUDGET)))
            hit = np.where(e < 1e-2)[0]
            circ.append(int(c[hit[0]]) if len(hit) else None); its.append(int(hit[0] + 1) if len(hit) else None)
            if sd == 0: curves[name] = (c, e)
        okc = [x for x in circ if x is not None]; oki = [x for x in its if x is not None]
        runs[name] = {"tuned_hyperparameters": hp, "error_at_4800_circuits_mean": st.mean(errs), "std": st.pstdev(errs),
                      "runs_reaching_error_1e-2": f"{len(okc)}/5", "iterations_to_1e-2_median": st.median(oki) if oki else None,
                      "circuit_executions_to_1e-2_median": st.median(okc) if okc else None}
    out["optimizers_L3_P16_tuned"] = runs
    out["note"] = ("circuit executions: parameter-shift = 2 per gate parameter (32 here); SPSA = 2 per step; QNG adds an ASSUMED 2P^2 = 512 "
                   "executions per step for the metric tensor (here computed from simulated states, so that count is an estimate)")
    # (c) the slide-28 QUBO as a Hamiltonian with a 12-parameter ansatz
    d = -qsim.maxcut_diag(4, EDGES4); ops4 = qopt.hea_ops(4, 2)
    f4 = lambda t: qsim.expval_diag(qsim.run(ops4, t, 4), d)
    res = [minimize(f4, np.random.default_rng(s).uniform(0, 2 * np.pi, 12), method="BFGS").fun for s in range(10)]
    out["slide28_vqe_on_4node_qubo"] = {"ground_energy": -4.0, "found_min": min(res), "runs_reaching_-4": f"{sum(r < -3.999 for r in res)}/10",
                                        "worst_run": max(res)}
    OUT["vqe"] = out
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, (c, e) in curves.items():
        ax.loglog(c, np.maximum(e, 1e-12), label=name)
    ax.set_xlabel("circuit executions"); ax.set_ylabel("energy error"); ax.legend(); ax.set_title("VQE, TFIM n=4, 16 parameters (seed 0)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_vqe_optimizers.png"), dpi=110); plt.close(fig)


# ================================================================== QAOA on the Track B instances
def exp_qaoa():
    inst = json.load(open(B_RES))
    rows, per = {}, {1: [], 2: [], 3: []}
    ps_ratio = {p: {} for p in per}
    t_all = {}
    for it in inst:
        n, edges, mc = it["n"], [tuple(e) for e in it["edges"]], it["max_cut"]
        d = qsim.maxcut_diag(n, edges)
        for p in per:
            t = time.perf_counter()
            th, e = qopt.optimize_qaoa(n, d, p, restarts=6, seed=it["seed"])
            t_all[(n, p)] = t_all.get((n, p), 0) + time.perf_counter() - t
            psi = qopt.qaoa_state_fast(n, d, th[:p], th[p:])
            prob_opt = float(np.sum(np.abs(psi[d == mc]) ** 2))
            rng = np.random.default_rng(1)
            best1000 = float(d[qsim.sample_bitstrings(psi, 1000, rng)].max())
            rand1000 = float(d[rng.integers(0, 2 ** n, size=1000)].max())          # uniformly random bitstrings
            per[p].append({"n": n, "ratio": e / mc, "p_opt": prob_opt, "random_ratio": float(d.mean()) / mc, "best_of_1000_shots": best1000 / mc, "random_best_of_1000": rand1000 / mc})
    for p in per:
        rows[f"p={p}"] = {"mean_E[cut]/maxcut": st.mean(r["ratio"] for r in per[p]), "min_E[cut]/maxcut": min(r["ratio"] for r in per[p]),
                          "mean_P(optimal)": st.mean(r["p_opt"] for r in per[p]),
                          "mean_best_of_1000_shots/maxcut": st.mean(r["best_of_1000_shots"] for r in per[p]),
                          "mean_best_of_1000_RANDOM_bitstrings/maxcut": st.mean(r["random_best_of_1000"] for r in per[p]),
                          "random_guess_ratio": st.mean(r["random_ratio"] for r in per[p])}
    by_n = {}
    for n in (6, 8, 10, 12):
        by_n[f"n={n}"] = {f"p={p}": st.mean(r["ratio"] for r in per[p] if r["n"] == n) for p in per}
    OUT["qaoa_maxcut"] = {"instances": len(inst), "classical_baseline": "brute-force optimum (Track B); simulated annealing also reaches it in ms (Track B)",
                          "summary": rows, "ratio_by_size": by_n,
                          "simulation_seconds_per_size_p3": {str(n): t_all[(n, 3)] for n in (6, 8, 10, 12)}}
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in per:
        ax.plot([6, 8, 10, 12], [by_n[f"n={n}"][f"p={p}"] for n in (6, 8, 10, 12)], "o-", label=f"QAOA p={p}")
    ax.axhline(st.mean(r["random_ratio"] for r in per[1]), color="gray", ls=":", label="random guess")
    ax.set_ylim(0.5, 1.0); ax.set_xlabel("nodes"); ax.set_ylabel("E[cut] / max cut"); ax.legend(); ax.set_title("QAOA on Track B Max-Cut graphs (exact simulation)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_qaoa.png"), dpi=110); plt.close(fig)


# ================================================================== finite shots
def exp_shots():
    inst = [i for i in json.load(open(B_RES)) if i["n"] == 8][0]
    n, edges, mc = 8, [tuple(e) for e in inst["edges"]], inst["max_cut"]
    d = qsim.maxcut_diag(n, edges); p = 2
    th_star, e_star = qopt.optimize_qaoa(n, d, p, restarts=8, seed=3)
    rng = np.random.default_rng(0)
    psi = qopt.qaoa_state_fast(n, d, th_star[:p], th_star[p:])
    var = float(np.sum(np.abs(psi) ** 2 * d ** 2) - e_star ** 2)
    est = {}
    for S in (10, 100, 1000, 10000):
        v = [qsim.sample_mean(psi, d, S, rng) for _ in range(400)]
        est[str(S)] = {"empirical_std": float(np.std(v)), "theory_sqrt(var/S)": float(np.sqrt(var / S))}
    rows = {}
    for S in (10, 100, 1000, 10000, None):
        finals, evals, starts = [], [], []
        for seed in range(6):
            r = np.random.default_rng(seed)
            if S is None:
                f = lambda t: -qsim.expval_diag(qopt.qaoa_state_fast(n, d, t[:p], t[p:]), d)
            else:
                f = lambda t, S=S, r=r: -qsim.sample_mean(qopt.qaoa_state_fast(n, d, t[:p], t[p:]), d, S, r)
            x0 = np.concatenate([np.random.default_rng(10 + seed).uniform(0, np.pi, p), np.random.default_rng(20 + seed).uniform(0, np.pi / 2, p)])
            th, h = qopt.spsa(f, x0, 150, a=0.5, c=0.2, seed=seed)
            exact = lambda t: qsim.expval_diag(qopt.qaoa_state_fast(n, d, t[:p], t[p:]), d) / mc      # re-evaluated EXACTLY
            finals.append(exact(th)); starts.append(exact(x0)); evals.append(h[-1][0])
        rows[str(S) if S else "exact (infinite shots)"] = {"final_exact_ratio_mean": st.mean(finals), "std": st.pstdev(finals),
                                                           "start_ratio_mean": st.mean(starts), "circuit_evals": st.mean(evals),
                                                           "total_shots": st.mean(evals) * S if S else None}
    OUT["finite_shots"] = {"graph": f"n={n}, p={p}", "noiseless_optimum_ratio": e_star / mc, "estimator_std_check": est, "spsa_150_iterations_with_shot_noise": rows}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    S = [10, 100, 1000, 10000]
    ax.semilogx(S, [rows[str(s)]["final_exact_ratio_mean"] for s in S], "o-", label="SPSA (150 it.), shot-noisy objective")
    ax.axhline(e_star / mc, color="k", ls="--", label="exact optimum"); ax.set_xlabel("shots per energy estimate"); ax.set_ylabel("exact ratio of found parameters")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_shots.png"), dpi=110); plt.close(fig)


# ================================================================== hardware noise (simulated)
def exp_noise():
    inst = [i for i in json.load(open(B_RES)) if i["n"] == 6][0]
    n, edges, mc = 6, [tuple(e) for e in inst["edges"]], inst["max_cut"]
    d = qsim.maxcut_diag(n, edges)
    rows = {}
    levels = (0.0, 0.001, 0.003, 0.01, 0.03)
    for p in (1, 2, 3):
        th, e = qopt.optimize_qaoa(n, d, p, restarts=6, seed=5)
        ops = qopt.qaoa_ops(n, edges, p)
        rows[f"p={p}"] = {"gates": len(ops), "ratios": {str(pd): float(qsim.run_dm(ops, th, n, pd) @ d / mc) for pd in levels}}
    OUT["noise_depolarizing"] = {"model": "single-qubit depolarizing channel after every gate on each qubit it touches (density-matrix simulation, simulated not hardware)",
                                 "graph": f"n={n}, {len(edges)} edges", "random_guess_ratio": float(d.mean() / mc), "rows": rows}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p in (1, 2, 3):
        ax.plot(levels, [rows[f"p={p}"]["ratios"][str(l)] for l in levels], "o-", label=f"p={p} ({rows[f'p={p}']['gates']} gates)")
    ax.axhline(d.mean() / mc, color="gray", ls=":", label="random guess"); ax.set_xlabel("depolarizing probability per gate")
    ax.set_ylabel("E[cut] / max cut"); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_noise.png"), dpi=110); plt.close(fig)


# ================================================================== barren plateaus
def exp_barren():
    samples = 150
    rows = {}
    for n in range(2, 12):
        for label, L, qubits in (("global cost, depth n", n, list(range(n))), ("local cost Z0 Z1, depth n", n, [0, 1]), ("local cost Z0 Z1, depth 2", 2, [0, 1])):
            if n < 2: continue
            ops = qopt.hea_ops(n, L); P = qopt.n_params(ops); d = qsim.z_string_diag(n, qubits)
            rng = np.random.default_rng(n * 100 + L)
            g = []
            for _ in range(samples):
                th = rng.uniform(0, 2 * np.pi, P)
                # gradient wrt the FIRST parameter only (2 circuit executions)
                fp = qsim.expval_diag(qsim.run(ops, th, n, {0: np.pi / 2}), d); fm = qsim.expval_diag(qsim.run(ops, th, n, {0: -np.pi / 2}), d)
                g.append(0.5 * (fp - fm))
            rows.setdefault(label, {})[n] = float(np.var(g))
    fits = {}
    for label, r in rows.items():
        ns = np.array(sorted(r)); v = np.array([r[n] for n in ns])
        fits[label] = float(np.polyfit(ns, np.log(v), 1)[0])        # slope of ln Var vs n
    OUT["barren_plateau"] = {"samples_per_point": samples, "variance_of_dC/dtheta_1": {k: {str(n): v for n, v in r.items()} for k, r in rows.items()},
                             "slope_of_ln_variance_vs_qubits": fits, "note": "slope -0.69 would mean variance halves with each added qubit"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, r in rows.items():
        ax.semilogy(list(r), list(r.values()), "o-", label=label)
    ax.set_xlabel("qubits n"); ax.set_ylabel("Var[dC/dtheta_1]"); ax.legend(fontsize=7); ax.set_title("Barren plateau (random HEA circuits)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_barren.png"), dpi=110); plt.close(fig)


# ================================================================== annealing-hardware embedding limits
def exp_embedding():
    sys.path.insert(0, os.path.join(HERE, "..", "B_discrete"))
    import qubo as qb
    out = {}
    # (a) clique embedding resources on a Chimera graph C_m (4m logical vertices, chains of length m+1)
    table = {}
    for m in (4, 8, 12, 16):
        n_log = 4 * m
        table[f"Chimera C{m} ({8 * m * m} physical qubits)"] = {"max_dense_logical_vars": n_log, "chain_length": m + 1,
                                                               "physical_qubits_used": n_log * (m + 1)}
    out["clique_embedding_chimera"] = {"rule": "K_{4m} embeds in Chimera C_m with chains of length m+1 (standard clique embedding)", "table": table,
                                       "consequence": "a fully connected QUBO of 65+ variables does not fit a 2048-qubit C16 Chimera; physical qubits needed grow ~ n^2/4"}
    # (b) chain strength trade-off on a dense Max-Cut QUBO
    n, L = 8, 3
    edges = qb.random_graph(n, 0.9, seed=11)
    Q = qb.maxcut_qubo(n, edges)
    opt, _ = qb.brute_force_qubo(Q)
    rows = {}
    scale = float(np.abs(Q).max())
    for strength in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        # replace each logical variable by a chain of L physical variables; logical couplings use one pair of chain members
        N = n * L
        Qp = np.zeros((N, N))
        for i in range(n):
            for a in range(L):
                Qp[i * L + a, i * L + a] += Q[i, i] / L
            for a in range(L - 1):                      # ferromagnetic chain: s*(x_a - x_b)^2 = s(x_a + x_b - 2 x_a x_b)
                u, v = i * L + a, i * L + a + 1
                Qp[u, u] += strength * scale; Qp[v, v] += strength * scale; Qp[u, v] -= strength * scale; Qp[v, u] -= strength * scale
        for i in range(n):
            for j in range(i + 1, n):
                if Q[i, j] != 0:
                    u, v = i * L + (j % L), j * L + (i % L)       # limited connectivity: each pair uses one specific chain member
                    Qp[u, v] += Q[i, j]; Qp[v, u] += Q[i, j]
        broken, quality = 0, []
        runs = 20
        for s in range(runs):
            _, xp = qb.sa_qubo(Qp, sweeps=400, T0=3 * (1 + strength) * scale, T1=0.05, seed=s)
            xp = xp.reshape(n, L)
            is_broken = bool((xp.min(axis=1) != xp.max(axis=1)).any())
            broken += is_broken
            xl = (xp.mean(axis=1) > 0.5).astype(float)         # majority vote unembedding
            quality.append(-qb.energy(Q, xl) / -opt if opt != 0 else 0)
        rows[f"chain_strength={strength}"] = {"runs_with_broken_chains": f"{broken}/{runs}", "mean_logical_energy/optimum": st.mean(quality)}
    out["chain_strength_tradeoff"] = {"logical_vars": n, "chain_length": L, "rows": rows,
                                      "note": "weak chains break; strong chains rescale away the problem couplings (the same energy-scale effect as the penalty weight in Track B)"}
    OUT["annealing_embedding"] = out


if __name__ == "__main__":
    allf = {f.__name__: f for f in (exp_vqe, exp_qaoa, exp_shots, exp_noise, exp_barren, exp_embedding)}
    path = os.path.join(RES, "summary.json")
    if len(sys.argv) > 1 and os.path.exists(path):
        OUT.update(json.load(open(path)))
    for name in (sys.argv[1:] or allf):
        t = time.perf_counter(); allf[name](); print(f"{name}: {time.perf_counter() - t:.1f}s", flush=True)
    with open(path, "w") as f:
        json.dump(OUT, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
