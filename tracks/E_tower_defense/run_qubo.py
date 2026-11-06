"""QUBO view of tower placement (bridge to Tracks B and C). Writes results/qubo.json.
(1) Small instance: 14 candidate cells, k=3 guns -> exact, simulated annealing and QAOA (p=1,2) on the Track C simulator.
(2) Full 70-cell map, gun-only: QUBO+SA placement versus greedy / ILP / annealing, all scored by the true simulator."""
import json
import os
import random
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "B_discrete"))
sys.path.insert(0, os.path.join(HERE, "..", "C_quantum_optimization"))
import qopt  # noqa: E402  (Track C)
import qsim  # noqa: E402
import qubo as qb  # noqa: E402  (Track B)

from td.maps import make_wave, snake_map  # noqa: E402
from td.optim import anneal, greedy, ilp  # noqa: E402
from td.qubo_tower import energies_all, tower_qubo  # noqa: E402
from td.sim import DEFAULT_TYPES, simulate  # noqa: E402
from td.surrogate import Surrogate  # noqa: E402

GUN = {"gun": DEFAULT_TYPES["gun"]}
out = {}
g = snake_map(15, 11)

# ---- (1) small instance: exact vs SA vs QAOA
k, n = 3, 14
cells_all = g.buildable()
_, _, _, v_all = tower_qubo(g, GUN["gun"], k, 0.0)
P = 1.2 * float(v_all.max())          # penalty must exceed the largest marginal value of one tower, or the constraint is ignored (Track B lesson)
top = [cells_all[i] for i in np.argsort(-v_all)[:n]]
Q, const, cells, v = tower_qubo(g, GUN["gun"], k, P, cells=top)
E = energies_all(Q, const)
e_min, e_mean = float(E.min()), float(E.mean())
sa_best = min((qb.sa_qubo(Q, sweeps=300, T0=3.0, T1=0.05, seed=s)[0] + const for s in range(10)))
qaoa = {}
for p in (1, 2, 3):
    scale = float(E.max() - E.min())
    d = -(E - E.min()) / scale                                                  # normalised so that angles are O(1); QAOA maximises <d>
    th, e_exp = qopt.optimize_qaoa(n, d, p, restarts=12, seed=1)
    psi = qopt.qaoa_state_fast(n, d, th[:p], th[p:])
    energy_exp = float(E.min() - e_exp * scale)                                 # e_exp = <d>; back to energy units <E>
    prob = np.abs(psi) ** 2
    feasible = np.array([bin(i).count("1") == k for i in range(2 ** n)])
    qaoa[f"p={p}"] = {"expected_energy": energy_exp, "approx_ratio_(mean-E)/(mean-Emin)": float((e_mean - energy_exp) / (e_mean - e_min)),
                      "P(optimal)": float(prob[E <= e_min + 1e-9].sum()), "P(feasible: exactly k towers)": float(prob[feasible].sum())}
out["small_instance"] = {"candidate_cells": n, "towers_k": k, "penalty_P": P, "exact_min_is_feasible": bool(bin(int(np.argmin(E))).count("1") == k), "exact_min_energy": e_min, "mean_energy_of_all_2^n_states": e_mean,
                         "SA_best_energy_10_runs": sa_best, "SA_reaches_exact": abs(sa_best - e_min) < 1e-9, "QAOA": qaoa,
                         "feasible_states_fraction": float(sum(bin(i).count("1") == k for i in range(2 ** n)) / 2 ** n),
                         "note": "QAOA simulated exactly on 14 qubits; the full 70-cell QUBO would need 70 qubits (2^70 amplitudes) so it cannot be simulated"}

# ---- (2) full map, gun-only comparison on the TRUE simulator
rows = {}
for budget in (30, 40):
    kk = budget // DEFAULT_TYPES["gun"].cost
    res = {"QUBO+SA": [], "greedy (surrogate)": [], "ILP (surrogate)": [], "annealing on simulator": [], "random": []}
    for ws in range(3):
        wave = make_wave(ws, 40)
        sur = Surrogate(g, GUN, wave)
        Q, const, cells, v = tower_qubo(g, GUN["gun"], kk, 1.2 * float(v_all.max()))
        best = min((qb.sa_qubo(Q, sweeps=400, T0=3.0, T1=0.05, seed=s) for s in range(10)), key=lambda z: z[0])
        pl = {cells[i]: "gun" for i in np.where(best[1] > 0.5)[0]}
        res["QUBO+SA"].append((simulate(g, pl, GUN, wave).damage, len(pl) == kk))
        res["greedy (surrogate)"].append((simulate(g, greedy(sur, GUN, budget), GUN, wave).damage, True))
        res["ILP (surrogate)"].append((simulate(g, ilp(sur, GUN, budget)[0], GUN, wave).damage, True))
        res["annealing on simulator"].append((simulate(g, anneal(g, GUN, wave, budget, evals=600, seed=0)[0], GUN, wave).damage, True))
        from td.baselines import random_placement
        res["random"].append((st.mean(simulate(g, random_placement(g, GUN, budget, random.Random(s)), GUN, wave).damage for s in range(50)), True))
    rows[f"budget {budget} ({kk} guns), mean damage over 3 waves"] = {m: {"damage": st.mean(x[0] for x in r), "feasible_in_all": all(x[1] for x in r)} for m, r in res.items()}
out["full_map_gun_only"] = rows
json.dump(out, open(os.path.join(HERE, "results", "qubo.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
