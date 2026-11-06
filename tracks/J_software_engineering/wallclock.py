"""End-to-end wall-clock cost of running the full suite versus the ILP-minimised suite (includes pytest start-up and imports).
Writes results/wallclock.json. Needs results/summary.json from optimize.py (the chosen tests are recomputed here from tests.json)."""
import json
import os
import statistics as st
import subprocess
import sys
import time

from collect import RES, nodeid_to_key, site_pkg
from optimize import ilp_cover

REL = "networkx/algorithms/shortest_paths"


def run(args, env=None):
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no"] + args, capture_output=True, text=True, env=env)
    return time.perf_counter() - t, r.stdout.strip().splitlines()[-1]


def main():
    d = json.load(open(os.path.join(RES, "tests.json")))
    tests = d["tests"]
    cover = {t: set(v["lines"]) for t, v in tests.items() if v["lines"]}
    cost = {t: max(tests[t]["time"], 1e-4) for t in cover}
    chosen, _ = ilp_cover(cover, cost, set().union(*cover.values()))
    target = os.path.join(site_pkg(REL), "tests")
    col = subprocess.run([sys.executable, "-m", "pytest", target, "--collect-only", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True).stdout
    ids = {nodeid_to_key(l): l for l in col.splitlines() if "::" in l}
    sel = [os.path.join(target, ids[k]) for k in chosen if k in ids]          # collected ids are relative to the tests folder
    out = {"selected_tests": len(chosen), "node_ids_resolved": len(sel), "full_suite_tests": len(ids)}
    full = [run([target]) for _ in range(3)]
    red = [run(sel) for _ in range(3)]
    sel_file = os.path.join(RES, "selected_tests.txt")
    with open(sel_file, "w") as f:
        f.write("\n".join(chosen))
    env = {**os.environ, "SELECT_TESTS": sel_file, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))}
    plug = [run([target, "-p", "select_plugin"], env) for _ in range(3)]
    out["reduced_via_plugin_wall_s"] = [round(t, 2) for t, _ in plug]
    out["plugin_last_line"] = plug[-1][1]
    out["wall_clock_speedup_plugin"] = st.mean(t for t, _ in full) / st.mean(t for t, _ in plug)
    out["full_suite_wall_s"] = [round(t, 2) for t, _ in full]
    out["reduced_suite_wall_s"] = [round(t, 2) for t, _ in red]
    out["full_last_line"], out["reduced_last_line"] = full[-1][1], red[-1][1]
    out["wall_clock_speedup_node_ids_on_command_line"] = st.mean(t for t, _ in full) / st.mean(t for t, _ in red)
    json.dump(out, open(os.path.join(RES, "wallclock.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
