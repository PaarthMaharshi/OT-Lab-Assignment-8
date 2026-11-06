"""Test-suite minimization and prioritization as optimization problems.

Minimization = weighted set cover:  min sum_t time_t x_t   s.t.  every line covered by the full suite is covered by >= 1 chosen test.
Methods: all tests (baseline), random order until covered, greedy cost-effectiveness, ILP (exact), each also with unit cost (fewest tests).
Cost of minimizing is measured with mutation testing: how many mutants killed by the FULL suite are still killed by the reduced suite.
Prioritization: order tests to detect faults early; metric APFD over the mutants.
Usage: python optimize.py   (needs results/tests.json and results/kills.json)
"""
import json
import os
import random
import statistics as st

from ortools.linear_solver import pywraplp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def greedy_cover(cover, cost, universe):
    left, chosen = set(universe), []
    while left:
        best = max(cover, key=lambda t: len(cover[t] & left) / cost[t])
        if not cover[best] & left:
            break
        chosen.append(best)
        left -= cover[best]
    return chosen


def ilp_cover(cover, cost, universe, time_limit_s=120):
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(int(time_limit_s * 1000))
    x = {t: s.BoolVar("") for t in cover}
    by_line = {}
    for t, ls in cover.items():
        for l in ls:
            by_line.setdefault(l, []).append(t)
    for l in universe:
        s.Add(sum(x[t] for t in by_line[l]) >= 1)
    s.Minimize(sum(cost[t] * x[t] for t in cover))
    status = s.Solve()
    return [t for t in cover if x[t].solution_value() > 0.5], status == pywraplp.Solver.OPTIMAL


def random_cover(cover, universe, seed):
    rng = random.Random(seed)
    order = list(cover)
    rng.shuffle(order)
    left, chosen = set(universe), []
    for t in order:
        if cover[t] & left:
            chosen.append(t); left -= cover[t]
        if not left:
            break
    return chosen


def evaluate(chosen, cover, cost, universe, kills, killable):
    cs = set(chosen)
    covered = set().union(*[cover[t] for t in cs]) if cs else set()
    killed = {i for i in killable if cs & set(kills[i])}
    return {"tests": len(cs), "runtime_s": sum(cost[t] for t in cs), "line_coverage_of_full_suite": len(covered & universe) / len(universe),
            "mutants_killed": len(killed), "mutation_score_vs_full_suite": len(killed) / len(killable) if killable else None}


def apfd(order, kills, killable):
    """Average percentage of faults detected: 1 - sum(first-detecting position)/(n*m) + 1/(2n)."""
    pos = {t: i + 1 for i, t in enumerate(order)}
    n, m = len(order), len(killable)
    s = sum(min(pos[t] for t in kills[i] if t in pos) for i in killable)
    return 1 - s / (n * m) + 1 / (2 * n)


def main():
    d = json.load(open(os.path.join(RES, "tests.json")))
    tests = d["tests"]
    cover = {t: set(v["lines"]) for t, v in tests.items() if v["lines"]}
    cost = {t: max(tests[t]["time"], 1e-4) for t in cover}
    universe = set().union(*cover.values())
    mut = json.load(open(os.path.join(RES, "kills.json")))
    kills = [[t for t in m["killed_by"] if t in cover] for m in mut]
    killable = [i for i, k in enumerate(kills) if k]
    out = {"package": d["package"], "source_lines_in_package_files": d["source_lines_of_code_files"], "tests_with_coverage": len(cover),
           "covered_lines": len(universe), "mutants_generated": len(mut), "mutants_killed_by_full_suite": len(killable),
           "mutation_sample_note": "operator-swap mutants on covered lines, seed 0; survivors of the full suite are excluded from the retention score"}
    res = {}
    res["full suite (baseline)"] = evaluate(list(cover), cover, cost, universe, kills, killable)
    unit = {t: 1.0 for t in cover}
    for label, c in (("time-weighted", cost), ("unit cost (fewest tests)", unit)):
        g = greedy_cover(cover, c, universe)
        res[f"greedy, {label}"] = evaluate(g, cover, cost, universe, kills, killable)
        i, optimal = ilp_cover(cover, c, universe)
        res[f"ILP, {label}"] = {**evaluate(i, cover, cost, universe, kills, killable), "proven_optimal": optimal}
    rs = [evaluate(random_cover(cover, universe, s), cover, cost, universe, kills, killable) for s in range(30)]
    res["random until covered (mean of 30)"] = {k: st.mean(r[k] for r in rs) for k in rs[0]}
    # fair comparison of fault detection: random subsets with the SAME number of tests as the ILP reduced suite
    k = res["ILP, time-weighted"]["tests"]
    rr = []
    for s in range(200):
        pick = random.Random(s).sample(list(cover), k)
        rr.append(evaluate(pick, cover, cost, universe, kills, killable))
    res[f"random subset of the same size ({k} tests), mean of 200"] = {kk: st.mean(r[kk] for r in rr) for kk in rr[0]}
    out["minimization"] = res
    # prioritization
    names = list(cover)
    orig = sorted(names)
    greedy_order, left, rest = [], set(universe), set(names)
    while rest:
        t = max(rest, key=lambda t: len(cover[t] & left) / cost[t])
        greedy_order.append(t); left -= cover[t]; rest.discard(t)
        if not left and rest:                       # coverage exhausted: remaining tests by cheapness
            greedy_order += sorted(rest, key=lambda t: cost[t]); break
    rnd = [apfd(random.Random(s).sample(names, len(names)), kills, killable) for s in range(100)]
    out["prioritization_APFD"] = {"original (alphabetical)": apfd(orig, kills, killable), "random (mean of 100)": st.mean(rnd),
                                  "greedy additional coverage per second": apfd(greedy_order, kills, killable)}
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
