"""Random-placement baseline on the default map. Writes results/baseline_random.json
and results/baseline_map.png. Usage: python run_baseline.py [budget] [runs]"""
import json
import os
import random
import statistics as st
import sys
import time

from td.baselines import random_placement
from td.maps import make_wave, snake_map
from td.sim import DEFAULT_TYPES, simulate, validate_placement
from td.surrogate import Surrogate
from td.viz import draw

HERE = os.path.dirname(os.path.abspath(__file__))
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30   # 120 saturates: every random layout kills everything
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 100

gmap = snake_map(15, 11)
wave = make_wave(seed=0, n=40)
sur = Surrogate(gmap, DEFAULT_TYPES, wave)
total_hp = sum(e.hp for e in wave)

rows, best = [], None
t0 = time.perf_counter()
for seed in range(runs):
    p = random_placement(gmap, DEFAULT_TYPES, budget, random.Random(seed))
    validate_placement(gmap, p, DEFAULT_TYPES, budget)
    r = simulate(gmap, p, DEFAULT_TYPES, wave)
    rows.append({"seed": seed, "damage": r.damage, "leaked": r.leaked,
                 "surrogate": sur.value(p), "towers": len(p)})
    if best is None or r.damage > best[0]:
        best = (r.damage, p)
elapsed = time.perf_counter() - t0

def stat(k):
    v = [x[k] for x in rows]
    return {"mean": st.mean(v), "std": st.pstdev(v), "min": min(v), "max": max(v)}

summary = {"map": "snake 15x11", "enemies": len(wave), "total_enemy_hp": total_hp,
           "budget": budget, "runs": runs, "sim_seconds_total": elapsed,
           "damage": stat("damage"), "leaked": stat("leaked"), "surrogate": stat("surrogate")}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "baseline_random.json"), "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=1)
draw(gmap, best[1], os.path.join(HERE, "results", "baseline_map.png"),
     f"best random placement (damage {best[0]:.0f}/{total_hp:.0f})")
print(json.dumps(summary, indent=1))
