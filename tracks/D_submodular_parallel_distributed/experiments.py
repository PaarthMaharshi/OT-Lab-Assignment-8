"""Track D experiments. Run: python experiments.py -> results/summary.json (+ figure).
Parallel results are MEASURED on this machine (multiprocessing); pipeline parallelism is simulated."""
import json
import os
import statistics as st
import time
import warnings
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dlib as d
import pwork

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = {}


def exp_submodular():
    rows = {}
    for k in (3, 5, 8):
        ratios = []
        for seed in range(30):
            sets = d.make_coverage(40, 150, 0.06, seed)
            ratios.append(d.coverage(sets, d.greedy_cover(sets, k)) / d.opt_cover(sets, k))
        rows[f"k={k}"] = {"greedy/optimum_mean": st.mean(ratios), "min": min(ratios), "bound_1-1/e": 1 - 1 / np.e,
                          "instances_where_greedy_is_not_optimal": sum(r < 1 - 1e-9 for r in ratios)}
    sets = d.make_coverage(30, 100, 0.1, 0)
    OUT["submodular"] = {"objective": "max coverage (monotone submodular), 30 instances per k", "rows": rows,
                         "diminishing_returns_violations_coverage": d.is_submodular_on_samples(lambda S: d.coverage(sets, S), 30),
                         "diminishing_returns_violations_|S|^2_supermodular": d.is_submodular_on_samples(lambda S: len(S) ** 2, 30),
                         "note": "the (1-1/e) bound is checked, not assumed: it needs monotone + submodular + cardinality constraint"}


def timeit(f, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


def run_mode(tasks, p, mode, pool):
    if mode == "static":
        pool.map(pwork.burn_block, [b.tolist() for b in np.array_split(np.array(tasks), p)], chunksize=1)
    elif mode == "dynamic":
        list(pool.imap_unordered(pwork.burn, tasks, chunksize=1))
    else:                                                       # LPT: biggest tasks first
        list(pool.imap_unordered(pwork.burn, sorted(tasks, reverse=True), chunksize=1))


def exp_parallel():
    rng = np.random.default_rng(0)
    irregular = [int(x) for x in rng.lognormal(mean=11.2, sigma=1.0, size=240)]
    uniform = [int(np.mean(irregular))] * 240
    ps = (1, 2, 4, 8, 12)
    out = {"cores_on_machine": os.cpu_count(), "tasks": 240}
    rows = {}
    for name, tasks in (("irregular", irregular), ("uniform", uniform)):
        t1 = timeit(lambda: [pwork.burn(n) for n in tasks], 1)
        res = {"serial_s": t1, "lower_bound_seconds": {}}
        for p in ps[1:]:
            with Pool(p) as pool:
                pool.map(pwork.burn, [10] * p * 2)              # warm-up: process start-up excluded
                for mode in ("static", "dynamic", "dynamic+LPT"):
                    tp = timeit(lambda: run_mode(tasks, p, mode, pool))
                    res.setdefault(mode, {})[str(p)] = {"seconds": tp, "speedup": t1 / tp, "efficiency": t1 / tp / p}
            res["lower_bound_seconds"][str(p)] = max(sum(tasks) / p, max(tasks)) / sum(tasks) * t1
        rows[name] = res
    out["load_balancing"] = rows
    s12 = rows["irregular"]["dynamic+LPT"]["12"]["speedup"]
    ser = (1 / s12 - 1 / 12) / (1 - 1 / 12)
    out["amdahl_fit"] = {"measured_speedup_p12_LPT": s12, "implied_serial_fraction": ser,
                         "model_speedups": {str(p): d.amdahl(max(ser, 0.0), p) for p in ps},
                         "caveat": "12 logical processors may be 6 physical cores with hyper-threading; the fit lumps that in"}
    tiny = [2000] * 6000
    t1 = timeit(lambda: [pwork.burn(n) for n in tiny], 1)
    ov = {"serial_s": t1}
    with Pool(8) as pool:
        pool.map(pwork.burn, [10] * 16)
        for cs in (1, 10, 100, 750):
            tp = timeit(lambda: list(pool.imap_unordered(pwork.burn, tiny, chunksize=cs)))
            ov[f"chunksize={cs}"] = {"seconds": tp, "speedup": t1 / tp}
    out["communication_overhead_tiny_tasks_p8"] = ov
    n, dim, steps = 2_000_000, 20, 5
    w = np.zeros(dim)
    pwork.init_data(n, dim, 0)
    t1 = timeit(lambda: [pwork.shard_grad((w, 0, n)) for _ in range(steps)], 2)
    dp = {"serial_s_per_5_steps": t1}
    for p in (2, 4, 8, 12):
        with Pool(p, initializer=pwork.init_data, initargs=(n, dim, 0)) as pool:
            bounds = np.linspace(0, n, p + 1).astype(int)
            jobs = [(w, int(bounds[i]), int(bounds[i + 1])) for i in range(p)]
            pool.map(pwork.shard_grad, jobs)
            tp = timeit(lambda: [sum(pool.map(pwork.shard_grad, jobs)) for _ in range(steps)], 2)
        dp[f"p={p}"] = {"seconds": tp, "speedup": t1 / tp, "efficiency": t1 / tp / p}
    out["data_parallel_gradient_2M_samples"] = dp
    rng = np.random.default_rng(1)
    sim = {}
    for p in (4, 8):
        res = []
        for _ in range(200):
            t = rng.lognormal(0, 1, 64).tolist()
            q = [t[i::p] for i in range(p)]
            res.append((d.makespan_static(t, p), d.makespan_static_queues(q), d.makespan_work_stealing(q), d.makespan_dynamic(t, p), sum(t) / p))
        r = np.mean(res, axis=0)
        sim[f"p={p}"] = {"static_block": r[0], "round_robin_queues": r[1], "work_stealing": r[2], "dynamic_pool": r[3], "lower_bound_total/p": r[4]}
    out["simulated_makespan_64_lognormal_tasks"] = sim
    out["pipeline_parallelism_simulated"] = {f"S=4,m={m}": {"simulated_bubble": 1 - m / d.pipeline_makespan(4, m), "formula": d.pipeline_bubble(4, m)}
                                             for m in (1, 4, 8, 32)}
    OUT["parallel"] = out
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mode in ("static", "dynamic", "dynamic+LPT"):
        ax[0].plot(ps[1:], [rows["irregular"][mode][str(p)]["speedup"] for p in ps[1:]], "o-", label=mode)
    ax[0].plot(ps, ps, "k:", label="ideal")
    ax[0].set_title("Irregular tasks: measured speedup")
    ax[0].set_xlabel("processes")
    ax[0].legend()
    ax[1].plot([2, 4, 8, 12], [dp[f"p={p}"]["speedup"] for p in (2, 4, 8, 12)], "s-", label="data-parallel gradient")
    ax[1].plot(ps, ps, "k:")
    ax[1].set_title("Data-parallel logistic gradient (2M samples)")
    ax[1].set_xlabel("processes")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig1_parallel.png"), dpi=110)
    plt.close(fig)


def exp_fedavg():
    out = {}
    for noniid in (False, True):
        clients, _ = d.make_clients(10, 200, 10, noniid=noniid, seed=0)
        _, hc = d.centralized_gd(clients, 3000, 0.5)
        fstar = hc[-1]
        rows = {"centralized GD (3000 steps)": {"final_loss": fstar}}
        for E in (1, 5, 20):
            _, h = d.fedavg(clients, 60, E, 0.1)
            hit = np.where(np.array(h) - fstar < 0.01)[0]
            rows[f"FedAvg E={E} local steps"] = {"loss_after_10_rounds": h[10], "loss_after_60_rounds": h[60],
                                                 "rounds_to_gap_0.01": int(hit[0]) if len(hit) else None, "gradient_steps_per_client": 60 * E}
        out["non-IID (feature shift + label skew)" if noniid else "IID"] = rows
    OUT["fedavg"] = out


def exp_multitask():
    res = {m: [] for m in ("equal", "mgda", "mgda_norm", "gradnorm")}
    for seed in range(5):
        prob = d.TwoTask(d=8, h=1, seed=seed)             # one shared feature: the tasks must compromise
        L0 = prob.losses(prob.init(seed))
        for m in res:
            res[m].append(d.train_multitask(prob, m, seed=seed) / L0)      # loss relative to the starting loss of each task
    OUT["multitask"] = {m: {"task1_final/initial": float(np.mean(v, axis=0)[0]), "task2_final/initial": float(np.mean(v, axis=0)[1]),
                            "mean_of_both": float(np.mean(v))} for m, v in res.items()}
    OUT["multitask"]["note"] = ("2 tasks sharing ONE hidden feature, 5 problems, 1500 steps, same lr; losses relative to each task's initial loss "
                                "(lower is better). Equal weighting is dominated by the larger-scale task; raw MGDA is dominated by the "
                                "smaller-gradient task; these are trade-offs, not a single winner")


if __name__ == "__main__":
    for fn in (exp_submodular, exp_fedavg, exp_multitask, exp_parallel):
        t = time.perf_counter()
        fn()
        print(fn.__name__, f"{time.perf_counter() - t:.1f}s", flush=True)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
