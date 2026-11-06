"""Track I experiments (SIMULATED; no real quantum hardware). Run: python experiments.py -> results/summary.json"""
import json
import math
import os
import statistics as st
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import qalgs as qa
import qml
import qsim

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = {}


def exp_algorithms():
    rng = np.random.default_rng(0)
    rows = {}
    dj = {}
    for n in (2, 4, 6, 8, 10):
        ok = 0
        for _ in range(20):
            if rng.random() < 0.5:
                f = np.full(2 ** n, int(rng.integers(0, 2))); expect = 1.0
            else:
                f = np.zeros(2 ** n, int); f[rng.choice(2 ** n, 2 ** (n - 1), replace=False)] = 1; expect = 0.0
            ok += abs(qa.deutsch_jozsa(n, f) - expect) < 1e-9
        dj[f"n={n}"] = {"correct_of_20": ok, "quantum_queries": 1, "classical_worst_case_queries": 2 ** (n - 1) + 1}
    rows["deutsch_jozsa"] = dj
    bv = {}
    for n in (2, 4, 8, 12):
        s = [int(b) for b in rng.integers(0, 2, n)]
        t = time.perf_counter(); got, p = qa.bernstein_vazirani(s); dt = time.perf_counter() - t
        bv[f"n={n}"] = {"recovered": got == s, "probability": p, "quantum_queries": 1, "classical_queries": n, "simulation_seconds": dt}
    rows["bernstein_vazirani"] = bv
    gr = {}
    for n in range(2, 15, 2):
        N = 2 ** n; k = qa.grover_optimal_iterations(n); marked = int(rng.integers(0, N))
        t = time.perf_counter(); p, calls = qa.grover(n, marked, k); tq = time.perf_counter() - t
        arr = list(range(N))
        t = time.perf_counter(); found = next(i for i, v in enumerate(arr) if v == marked); tc = time.perf_counter() - t
        gr[f"n={n} (N={N})"] = {"success_probability": p, "oracle_queries_quantum": calls, "classical_expected_queries": (N + 1) / 2, "classical_worst_queries": N,
                                "query_ratio_classical_expected/quantum": (N + 1) / 2 / calls, "simulation_seconds": tq, "classical_python_linear_search_seconds": tc}
    rows["grover"] = gr
    # success probability versus iteration count shows the over-rotation
    n = 6; curve = [qa.grover(n, 5, k)[0] for k in range(0, 21)]
    rows["grover_overrotation_n6"] = {"iterations_0_to_20": curve, "optimal_k": qa.grover_optimal_iterations(n), "peak_at": int(np.argmax(curve))}
    rows["qft"] = {f"t={t}": {"max_abs_difference_to_DFT_matrix": float(np.abs(qa.qft_matrix(t) - np.array([[np.exp(2j * np.pi * j * k / 2 ** t) for k in range(2 ** t)] for j in range(2 ** t)]) / math.sqrt(2 ** t)).max()),
                              "gates": t * (t + 1) // 2 + t // 2} for t in (2, 4, 6, 8)}
    rows["qpe"] = {"phi=1/8, t=3": {"P(exact estimate)": float(qa.qpe(1 / 8, 3)[1])},
                   "phi=0.3, t=6": {"best_estimate": int(np.argmax(qa.qpe(0.3, 6))) / 64, "P(best)": float(qa.qpe(0.3, 6).max()), "textbook_lower_bound_4/pi^2": 4 / math.pi ** 2}}
    shor = {}
    srng = np.random.default_rng(1)
    for a in (2, 4, 7, 8, 11, 13):
        p = qa.shor_order_finding(a, 15, 8)
        shots = srng.choice(len(p), size=200, p=p / p.sum())
        found = [qa.order_from_measurement(int(m), 8, a) for m in shots]
        good = sum(1 for r in found if r == qa.classical_order(a))
        factor_ok = 0
        for r in found:
            if r and r % 2 == 0 and pow(a, r // 2, 15) != 14:
                g = math.gcd(pow(a, r // 2) - 1, 15)
                factor_ok += g in (3, 5)
        shor[f"a={a}"] = {"true_order": qa.classical_order(a), "shots_with_correct_order": f"{good}/200", "shots_giving_a_nontrivial_factor": f"{factor_ok}/200"}
    rows["shor_N15"] = {"results": shor, "note": "strict decoding (no multiples of r are tried); about half the shots are uninformative by design (m=0 or half-period), which is why the success rate is near 1/2 and the algorithm is repeated. Modular exponentiation applied as exact permutations (not gate-decomposed); N=15 is trivial classically; no advantage is claimed or measurable at this size"}
    inst = [i for i in json.load(open(os.path.join(HERE, "..", "B_discrete", "results", "maxcut_instances.json"))) if i["n"] == 8][0]
    edges = [tuple(e) for e in inst["edges"]]
    ann = {}
    for T in (0.5, 2, 8, 30, 100):
        pr, gap = qa.anneal(8, edges, T)
        ann[f"T={T}"] = {"P(optimal cut)": pr, "min_gap_along_schedule": gap}
    rows["quantum_annealing_simulated_n8"] = {"note": "closed-system adiabatic evolution, linear schedule; not a real annealer (no noise, no embedding)", **ann}
    OUT["algorithms"] = rows
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(21), curve, "o-"); ax[0].set_xlabel("Grover iterations"); ax[0].set_ylabel("P(marked)"); ax[0].set_title("Grover n=6: over-rotation after the optimum")
    ns = list(range(2, 15, 2))
    ax[1].loglog([2 ** n for n in ns], [gr[f"n={n} (N={2 ** n})"]["oracle_queries_quantum"] for n in ns], "o-", label="Grover queries")
    ax[1].loglog([2 ** n for n in ns], [(2 ** n + 1) / 2 for n in ns], "s--", label="classical expected queries")
    ax[1].set_xlabel("N"); ax[1].legend(); ax[1].set_title("Query counts")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_algorithms.png"), dpi=110); plt.close(fig)


def exp_noise():
    """Hardware noise model (depolarizing, density matrix): Bernstein-Vazirani with the all-ones secret."""
    rows = {}
    for n in (2, 3, 4, 5, 6):
        ops = [("x", (n,), None, 0)] + [("h", (q,), None, 0) for q in range(n + 1)] + [("cnot", (i, n), None, 0) for i in range(n)] + [("h", (q,), None, 0) for q in range(n)]
        r = {}
        for p in (0.0, 0.001, 0.005, 0.01, 0.03):
            probs = qsim.run_dm(ops, [], n + 1, p)
            r[str(p)] = float(sum(probs[(2 ** n - 1) * 2 + a] for a in (0, 1)))   # all-ones on inputs, ancilla anything
        rows[f"n={n} ({len(ops)} gates, {n} two-qubit)"] = r
    OUT["noise_bernstein_vazirani"] = {"model": "single-qubit depolarizing noise after each gate on each touched qubit (simulated, not hardware)", "success_probability": rows,
                                       "classical_one_query_per_bit_needs_n_queries_noise_free": True}


def split(name, seed):
    X, y = (make_moons(300, noise=0.2, random_state=seed) if name == "moons" else make_circles(300, noise=0.1, factor=0.5, random_state=seed))
    X = MinMaxScaler((0.1, math.pi - 0.1)).fit_transform(X)
    return train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)


def exp_qml():
    out = {}
    for ds in ("moons", "circles"):
        res = {k: {"acc": [], "train_s": [], "circuits": [], "params": []} for k in ("variational classifier (3 layers)", "quantum kernel + SVM", "logistic regression", "RBF SVM", "MLP (8 hidden)", "MLP (16,16)")}
        for seed in range(5):
            Xtr, Xte, ytr, yte = split(ds, seed)
            ypm = 2.0 * ytr - 1
            p, secs, runs = qml.train_vqc(Xtr, ypm, layers=3, steps=60, lr=0.1, seed=seed)
            r = res["variational classifier (3 layers)"]
            r["acc"].append(qml.vqc_accuracy(Xte, 2.0 * yte - 1, p, 3)); r["train_s"].append(secs); r["circuits"].append(runs); r["params"].append(len(p))
            t = time.perf_counter(); K = qml.kernel_matrix(Xtr, Xtr); clf = SVC(kernel="precomputed", C=10).fit(K, ytr); secs = time.perf_counter() - t
            r = res["quantum kernel + SVM"]
            r["acc"].append(clf.score(qml.kernel_matrix(Xte, Xtr), yte)); r["train_s"].append(secs)
            r["circuits"].append(len(Xtr) + len(Xtr) * (len(Xtr) - 1) // 2); r["params"].append(len(clf.support_) + 1)
            for name, model in (("logistic regression", LogisticRegression()), ("RBF SVM", SVC(C=10)), ("MLP (8 hidden)", MLPClassifier((8,), max_iter=2000, random_state=seed)), ("MLP (16,16)", MLPClassifier((16, 16), max_iter=3000, random_state=seed))):
                t = time.perf_counter(); model.fit(Xtr, ytr); secs = time.perf_counter() - t
                r = res[name]; r["acc"].append(model.score(Xte, yte)); r["train_s"].append(secs); r["circuits"].append(0)
                r["params"].append(3 if name == "logistic regression" else len(model.support_) * 3 if name == "RBF SVM" else 2 * 8 + 8 + 8 + 1 if name == 'MLP (8 hidden)' else 2 * 16 + 16 + 16 * 16 + 16 + 16 + 1)
        out[ds] = {k: {"test_acc_mean": st.mean(v["acc"]), "test_acc_std": st.pstdev(v["acc"]), "train_seconds": st.mean(v["train_s"]),
                       "circuit_evaluations": st.mean(v["circuits"]), "parameters": st.mean(v["params"])} for k, v in res.items()}
    OUT["qml_vs_classical"] = {"note": "300 points, 50/50 split, 5 seeds, 2 features; quantum models simulated on 2 qubits", **out}


if __name__ == "__main__":
    allf = {f.__name__: f for f in (exp_algorithms, exp_noise, exp_qml)}
    path = os.path.join(RES, "summary.json")
    if len(sys.argv) > 1 and os.path.exists(path):
        OUT.update(json.load(open(path)))
    for name in (sys.argv[1:] or allf):
        t = time.perf_counter(); allf[name](); print(name, f"{time.perf_counter() - t:.1f}s", flush=True)
    with open(path, "w") as f:
        json.dump(OUT, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
