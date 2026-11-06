"""Fair comparison: random vs greedy vs ILP vs simulated annealing.

Every placement is scored by the SAME true simulator on the SAME map/wave.
Instances: snake 15x11, wave seeds x budgets. Stochastic methods use repeated seeds.
Annealing gets a fixed number of simulator evaluations (SA_EVALS).
Usage: python compare.py
"""
import json
import os
import random
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from td.baselines import random_placement
from td.maps import make_wave, snake_map, spiral_map
from td.optim import anneal, greedy, ilp
from td.sim import DEFAULT_TYPES as TYPES, simulate, validate_placement
from td.surrogate import Surrogate

HERE = os.path.dirname(os.path.abspath(__file__))
WAVE_SEEDS = [0, 1, 2]
BUDGETS = [20, 30]   # 40+ saturates: good layouts kill every enemy, nothing left to compare
SA_SEEDS = [0, 1, 2, 3, 4]
SA_EVALS = 800
RANDOM_RUNS = 100


MAPS = {"snake 15x11": lambda: snake_map(15, 11), "spiral 13x11": lambda: spiral_map(13, 11)}


def instance(wave_seed, map_name="snake 15x11"):
    return MAPS[map_name](), make_wave(wave_seed, 40)


def sa_job(args):
    map_name, ws, budget, seed = args
    g, wave = instance(ws, map_name)
    t = time.perf_counter()
    p, _ = anneal(g, TYPES, wave, budget, evals=SA_EVALS, seed=seed)
    dt = time.perf_counter() - t
    validate_placement(g, p, TYPES, budget)
    r = simulate(g, p, TYPES, wave)
    return map_name, ws, budget, seed, r.damage, r.leaked, dt, p


def main():
    jobs = [(m, ws, b, s) for m in MAPS for ws in WAVE_SEEDS for b in BUDGETS for s in SA_SEEDS]
    with ProcessPoolExecutor() as ex:
        sa = list(ex.map(sa_job, jobs))
    out = []
    for map_name in MAPS:
      for ws in WAVE_SEEDS:
        g, wave = instance(ws, map_name)
        total = sum(e.hp for e in wave)
        sur = Surrogate(g, TYPES, wave)
        for b in BUDGETS:
            row = {"map": map_name, "wave_seed": ws, "budget": b, "total_hp": total}
            rs = []
            for s in range(RANDOM_RUNS):
                p = random_placement(g, TYPES, b, random.Random(s))
                r = simulate(g, p, TYPES, wave)
                rs.append((r.damage, r.leaked))
            row["random"] = {"damage_mean": st.mean(d for d, _ in rs),
                             "damage_std": st.pstdev(d for d, _ in rs),
                             "leaked_mean": st.mean(l for _, l in rs)}
            t = time.perf_counter()
            pg = greedy(sur, TYPES, b)
            tg = time.perf_counter() - t
            validate_placement(g, pg, TYPES, b)
            rg = simulate(g, pg, TYPES, wave)
            row["greedy"] = {"damage": rg.damage, "leaked": rg.leaked, "seconds": tg,
                             "surrogate": sur.value(pg), "towers": len(pg)}
            t = time.perf_counter()
            pi, opt = ilp(sur, TYPES, b)
            ti = time.perf_counter() - t
            validate_placement(g, pi, TYPES, b)
            ri = simulate(g, pi, TYPES, wave)
            row["ilp"] = {"damage": ri.damage, "leaked": ri.leaked, "seconds": ti,
                          "surrogate": sur.value(pi), "proven_optimal": opt,
                          "towers": len(pi)}
            mine = [x for x in sa if x[0] == map_name and x[1] == ws and x[2] == b]
            dm = [x[4] for x in mine]
            row["anneal"] = {"damage_mean": st.mean(dm), "damage_std": st.pstdev(dm),
                             "damage_best": max(dm), "leaked_mean": st.mean(x[5] for x in mine),
                             "seconds_mean": st.mean(x[6] for x in mine),
                             "evals": SA_EVALS, "runs": len(mine)}
            out.append(row)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "comparison.json"), "w") as f:
        json.dump(out, f, indent=1)

    lines = ["| map | wave | budget | total HP | random (mean±sd) | greedy | ILP | annealing (mean±sd, best) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(
            f"| {r['map']} | {r['wave_seed']} | {r['budget']} | {r['total_hp']:.0f} "
            f"| {r['random']['damage_mean']:.0f}±{r['random']['damage_std']:.0f} "
            f"| {r['greedy']['damage']:.0f} ({r['greedy']['seconds']*1000:.0f} ms) "
            f"| {r['ilp']['damage']:.0f} ({r['ilp']['seconds']:.1f} s) "
            f"| {r['anneal']['damage_mean']:.0f}±{r['anneal']['damage_std']:.0f}, {r['anneal']['damage_best']:.0f} "
            f"({r['anneal']['seconds_mean']:.1f} s) |")
    table = "\n".join(lines)
    with open(os.path.join(HERE, "results", "comparison.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    labels = [f"{r['map'].split()[0][:2]}/w{r['wave_seed']}/B{r['budget']}" for r in out]
    xs = range(len(out))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.2
    series = [("random", [r["random"]["damage_mean"] for r in out]),
              ("greedy", [r["greedy"]["damage"] for r in out]),
              ("ILP", [r["ilp"]["damage"] for r in out]),
              ("annealing", [r["anneal"]["damage_mean"] for r in out])]
    for k, (name, vals) in enumerate(series):
        ax.bar([x + (k - 1.5) * w for x in xs], vals, w, label=name)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("true simulated damage")
    ax.legend()
    ax.set_title("Placement methods, scored by the true simulator")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "comparison.png"), dpi=120)


if __name__ == "__main__":
    main()
