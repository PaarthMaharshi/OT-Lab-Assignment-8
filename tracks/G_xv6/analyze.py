"""Summarise results/bench.json into results/summary.json (paired within each boot: all three policies run in every boot)."""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "bench.json")))
NAMES = {0: "round robin", 1: "lottery", 2: "MLFQ"}


def rows(wl, pol, kind, rep=None):
    return [r for x in R if x["workload"] == wl and x["policy"] == pol and (rep is None or x["rep"] == rep) for r in x["rows"] if r["_kind"] == kind]


out = {"share": {}, "mixed": {}, "jobs": {}}
for pol in (0, 1, 2):
    by = {}
    for rep in range(3):
        rs = rows("share", pol, "share", rep)
        tot = sum(int(r["iters_k"]) for r in rs)
        for r in rs:
            by.setdefault(int(r["tickets"]), []).append(int(r["iters_k"]) / tot)
    ideal = {30: 0.5, 20: 1 / 3, 10: 1 / 6}
    out["share"][NAMES[pol]] = {f"tickets={t}": {"cpu_share_mean": st.mean(v), "std": st.pstdev(v)} for t, v in sorted(by.items(), reverse=True)}
    out["share"][NAMES[pol]]["max_abs_error_vs_ticket_proportional"] = max(abs(st.mean(v) - ideal[t]) for t, v in by.items())
    m = rows("mixed", pol, "mixed")
    out["mixed"][NAMES[pol]] = {"interactive_iterations": [int(r["interactive_iterations"]) for r in m],
                                "mean_wakeup_delay_ticks": [int(r["mean_wakeup_delay_x100"]) / 100 for r in m],
                                "max_wakeup_delay_ticks": [int(r["max_delay"]) for r in m]}
    j = rows("jobs", pol, "jobs_summary")
    out["jobs"][NAMES[pol]] = {"mean_turnaround_ticks": [int(r["mean_turnaround_x10"]) / 10 for r in j],
                               "mean_response_ticks": [int(r["mean_response_x10"]) / 10 for r in j], "makespan_ticks": [int(r["makespan"]) for r in j]}
# paired comparison against round robin within each boot
for pol in (1, 2):
    ratios = []
    for rep in range(3):
        a = rows("jobs", 0, "jobs_summary", rep)[0]; b = rows("jobs", pol, "jobs_summary", rep)[0]
        ratios.append(int(b["mean_turnaround_x10"]) / int(a["mean_turnaround_x10"]))
    out["jobs"][NAMES[pol]]["turnaround_vs_round_robin_same_boot"] = ratios
json.dump(out, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
