"""Track F experiments. Run: python experiments.py -> results/summary.json
C code is compiled with gcc -O0 so that GCC does not repeat our optimizations; we measure OUR passes."""
import itertools
import json
import os
import statistics as st
import subprocess
import time

import passes as P
import programs as pr
import toyc

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BUILD = os.path.join(RES, "build")
os.makedirs(BUILD, exist_ok=True)


def build_and_run(ir, tag, reps=7):
    c = os.path.join(BUILD, tag + ".c")
    exe = os.path.join(BUILD, tag + ".exe")
    code = toyc.to_c(ir)
    with open(c, "w") as f:
        f.write(code)
    subprocess.run(["gcc", "-O0", "-w", c, "-o", exe], check=True)
    subprocess.run([exe], capture_output=True, check=True)          # warm-up: first run of a new exe is slow (cold start)
    times, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.split()
        times.append(time.perf_counter() - t)
    return {"seconds": min(times), "c_bytes": len(code), "exe_bytes": os.path.getsize(exe)}, [int(x) for x in out]


def main():
    summary = {"note": "gcc -O0; runtime = best of 7 after a warm-up; instruction counts from the IR; all variants verified to print identical output"}
    rows = {}
    for name in pr.PROGRAMS:
        n = pr.SIZES[name]
        ir = toyc.compile_to_ir(pr.source(name, n))
        small = toyc.compile_to_ir(pr.source(name, pr.SMALL[name]))
        variants = {"unoptimized": lambda x: x, "cprop": lambda x: P.apply(x, ["cprop"]), "strength": lambda x: P.apply(x, ["strength"]),
                    "cse": lambda x: P.apply(x, ["cse"]), "dce": lambda x: P.apply(x, ["dce"]), "all (fixpoint)": P.fixpoint}
        ref_out = None
        r = {}
        for vname, fn in variants.items():
            t = time.perf_counter(); o = fn(ir); ct = time.perf_counter() - t
            dyn_ref = toyc.run_vm(fn(small))              # dynamic count measured on a small input (VM is slow)
            res, out = build_and_run(o, f"{name}_{vname.split()[0]}")
            if ref_out is None:
                ref_out = out
            assert out == ref_out, f"{name}/{vname}: output differs"          # correctness check
            assert dyn_ref[0] == toyc.run_vm(small)[0]
            r[vname] = {"static_instructions": P.static_count(o), "dynamic_instructions_small_input": dyn_ref[1], "compile_ms": ct * 1000, **res}
        resB, _ = build_and_run(ir, f"{name}_unoptimizedB")             # A/A control: identical code, built and timed again
        r["_noise_floor_AA_ratio"] = r["unoptimized"]["seconds"] / resB["seconds"]
        rows[name] = r
    summary["programs"] = rows
    gain = {name: {"noise_floor_AA_ratio": r["_noise_floor_AA_ratio"], "runtime_speedup": r["unoptimized"]["seconds"] / r["all (fixpoint)"]["seconds"],
                   "dynamic_instruction_reduction_pct": 100 * (1 - r["all (fixpoint)"]["dynamic_instructions_small_input"] / r["unoptimized"]["dynamic_instructions_small_input"]),
                   "static_reduction_pct": 100 * (1 - r["all (fixpoint)"]["static_instructions"] / r["unoptimized"]["static_instructions"]),
                   "code_size_change_pct": 100 * (r["all (fixpoint)"]["exe_bytes"] / r["unoptimized"]["exe_bytes"] - 1)} for name, r in rows.items()}
    summary["gain_all_passes"] = gain
    # ---- decision 1: pass ordering (4 passes, each applied once) as a search problem
    order_rows = {}
    for name in pr.PROGRAMS:
        ir = toyc.compile_to_ir(pr.source(name, 10))
        results = {order: P.static_count(P.apply(ir, order)) for order in itertools.permutations(P.PASSES)}
        best = min(results.values()); worst = max(results.values())
        # greedy: repeatedly apply the not-yet-used pass that shrinks the code most
        cur, left, order, evals = ir, list(P.PASSES), [], 0
        while left:
            sc = {p: P.static_count(P.apply(cur, [p])) for p in left}
            evals += len(left)
            p = min(sc, key=sc.get); cur = P.apply(cur, [p]); left.remove(p); order.append(p)
        order_rows[name] = {"unoptimized": P.static_count(ir), "best_of_24_orders": best, "worst_of_24_orders": worst,
                            "greedy_order": order, "greedy_result": P.static_count(cur), "greedy_equals_optimum": P.static_count(cur) == best,
                            "fixpoint_result": P.static_count(P.fixpoint(ir)), "evaluations_exhaustive": 24 * 4, "evaluations_greedy": evals,
                            "orders_reaching_best": sum(v == best for v in results.values())}
    summary["pass_ordering"] = order_rows
    # ---- decision 2: register allocation as graph colouring (greedy vs exact ILP)
    reg = {}
    for name in pr.PROGRAMS:
        ir = toyc.compile_to_ir(pr.source(name, 10))
        row = {}
        for label, code in (("unoptimized", ir), ("optimized", P.fixpoint(ir))):
            nodes, edges = P.interference(code)
            g = P.color_greedy(nodes, edges)
            t = time.perf_counter(); e = P.color_exact(nodes, edges, g); dt = time.perf_counter() - t
            row[label] = {"variables": len(nodes), "interference_edges": len(edges), "registers_greedy": g, "registers_exact_ILP": e, "ilp_seconds": dt}
        reg[name] = row
    summary["register_allocation"] = reg
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("gain_all_passes", "pass_ordering", "register_allocation")}, indent=1))


if __name__ == "__main__":
    main()
