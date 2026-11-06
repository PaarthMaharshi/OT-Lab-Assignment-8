"""Link to Track D: annealing runs are independent, so evaluation parallelises across processes.
Measures wall time of 12 identical annealing jobs with 1, 2, 4, 6, 12 worker processes. Writes results/parallel_eval.json."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from compare import sa_job

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    jobs = [("snake 15x11", 0, 30, s) for s in range(12)]
    rows, t1 = {}, None
    for w in (1, 2, 4, 6, 12):
        t = time.perf_counter()
        with ProcessPoolExecutor(w) as ex:
            res = list(ex.map(sa_job, jobs))
        dt = time.perf_counter() - t
        t1 = t1 or dt
        rows[f"workers={w}"] = {"seconds": dt, "speedup": t1 / dt, "efficiency": t1 / dt / w, "mean_damage": sum(r[4] for r in res) / len(res)}
        print(w, rows[f"workers={w}"], flush=True)
    json.dump({"jobs": 12, "task": "annealing, 800 simulator evaluations each", "rows": rows,
               "note": "process start-up included; 12 logical processors on this machine"}, open(os.path.join(HERE, "results", "parallel_eval.json"), "w"), indent=1)
