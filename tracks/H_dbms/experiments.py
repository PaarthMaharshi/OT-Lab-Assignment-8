"""Track H experiments. Run: python experiments.py -> results/summary.json
All latencies are measured on SQLite (python sqlite3) with the default planner; results of every plan are compared
with the baseline result sets (must be identical)."""
import itertools
import json
import os
import shutil
import time

import hlib as h

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


WORK = os.path.join(RES, "build", "work.db")


def fresh():
    """Connection to a fresh copy of the index-free baseline database (so index sizes and timings are isolated)."""
    shutil.copyfile(h.DB, WORK)
    return h.connect(WORK)


def with_indexes(keys):
    con = fresh()
    for k in keys:
        h.create_index(con, k)
    if keys:
        con.execute("ANALYZE")                              # refresh planner statistics after creating indexes
        con.commit()
    return con


def plan(con, sql):
    return [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()]


def main():
    t = time.perf_counter()
    h.build_db()
    print(f"database ready ({time.perf_counter() - t:.1f}s)", flush=True)
    con = h.connect()
    out = {"dataset": h.SIZES, "dbms": "SQLite " + con.execute("select sqlite_version()").fetchone()[0], "workload_queries": len(h.WORKLOAD)}
    base_bytes = h.db_pages(con)
    base_lat, base_res = h.workload_latency(con)
    out["baseline"] = {"latency_s": base_lat, "weighted_total_s": h.weighted(base_lat), "db_mb": base_bytes / 1e6,
                       "plans": {q: plan(con, s) for q, (w, s) in h.WORKLOAD.items()}}
    con.close()
    print("baseline total", out["baseline"]["weighted_total_s"], flush=True)
    # ---- measure every candidate index alone, on its own fresh copy of the database
    size, bq, cand_rows = {}, {q: {} for q in h.WORKLOAD}, {}
    for key in h.CANDIDATES:
        t = time.perf_counter()
        con = with_indexes([key]); build = time.perf_counter() - t
        size[key] = (h.db_pages(con) - base_bytes) / 1e6
        lat, res = h.workload_latency(con)
        for q in h.WORKLOAD:
            assert res[q] is None or res[q] == base_res[q], f"{key}/{q}: result differs"
            bq[q][key] = max(0.0, base_lat[q] - lat[q])
        cand_rows[key] = {"definition": h.CANDIDATES[key], "size_mb": size[key], "build_and_analyze_s": build, "saving_s": sum(bq[q][key] for q in h.WORKLOAD),
                          "weighted_total_after_s": h.weighted(lat)}
        con.close()
        print("candidate", key, round(size[key], 1), "MB, saving", round(cand_rows[key]["saving_s"], 2), flush=True)
    out["candidates"] = cand_rows
    total = {k: sum(bq[q][k] for q in h.WORKLOAD) for k in size}
    total_size = sum(size.values())
    out["total_candidate_size_mb"] = total_size
    # ---- selection under a storage budget
    sel_rows = {}
    for frac in (0.15, 0.3, 0.6):
        B = frac * total_size
        methods = {"greedy (benefit/size, additive model)": h.select_greedy(total, size, B), "DP knapsack (additive model)": h.select_dp(total, size, B),
                   "ILP (best index per query)": h.select_ilp(bq, size, B)}
        row = {"budget_mb": B}
        for name, chosen in methods.items():
            con = with_indexes(chosen)
            lat, res = h.workload_latency(con)
            assert all(res[q] is None or res[q] == base_res[q] for q in h.WORKLOAD), f"{name}: result differs"
            row[name] = {"indexes": chosen, "storage_mb": (h.db_pages(con) - base_bytes) / 1e6, "weighted_total_s": h.weighted(lat),
                         "actual_saving_s": h.weighted(base_lat) - h.weighted(lat), "queries_timed_out_30s": [q for q in h.WORKLOAD if res[q] is None], "additive_model_predicted_s": sum(total[k] for k in chosen),
                         "best-index-per-query_model_predicted_s": sum(max([bq[q][k] for k in chosen] + [0]) for q in h.WORKLOAD)}
            con.close()
        sel_rows[f"budget={int(frac * 100)}% of all candidates"] = row
        print("budget", frac, "done", flush=True)
    con = with_indexes(list(h.CANDIDATES))
    lat_all, res_all = h.workload_latency(con)
    assert all(res_all[q] is None or res_all[q] == base_res[q] for q in h.WORKLOAD)
    sel_rows["all 15 candidate indexes"] = {"queries_timed_out_30s": [q for q in h.WORKLOAD if res_all[q] is None], "storage_mb": (h.db_pages(con) - base_bytes) / 1e6, "weighted_total_s": h.weighted(lat_all),
                                            "actual_saving_s": h.weighted(base_lat) - h.weighted(lat_all)}
    out["index_selection"] = sel_rows
    out["plans_with_all_indexes"] = {q: plan(con, h.WORKLOAD[q][1]) for q in ("Q3_orders_lines", "Q5_four_way")}
    # ---- join order for the 4-way query (all indexes present so the order matters)
    base = h.filtered_cards(con)
    dp_order, greedy_order = h.join_order_dp(base), h.join_order_greedy(base)
    default_t, default_rows = h.run_query(con, h.WORKLOAD["Q5_four_way"][1], reps=3)
    timing = {}
    for order in itertools.permutations(h.TABLES):
        tt, rows = h.run_query(con, h.forced_sql(order), reps=1, timeout_s=20)
        if rows is not None:
            assert rows == default_rows, f"order {order} gives a different result"
        timing["->".join(order)] = {"seconds": tt, "estimated_C_out": h.order_cost(list(order), base)}
        print("order", order, tt, flush=True)
    done = {k: v["seconds"] for k, v in timing.items() if v["seconds"] is not None}
    best_name = min(done, key=done.get)
    fmt = lambda o: "->".join(o)
    out["join_order"] = {"filtered_cardinalities": base, "sqlite_default_planner_s": default_t, "best_measured_order": best_name, "best_measured_s": done[best_name],
                         "worst_completed_s": max(done.values()), "orders_timed_out_20s": [k for k, v in timing.items() if v["seconds"] is None],
                         "dp_order": fmt(dp_order), "dp_order_s": timing[fmt(dp_order)]["seconds"], "greedy_order": fmt(greedy_order),
                         "greedy_order_s": timing[fmt(greedy_order)]["seconds"], "all_orders": timing}
    con.close()
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
