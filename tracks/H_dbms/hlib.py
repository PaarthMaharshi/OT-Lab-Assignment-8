"""Track H library: synthetic database, workload, index selection (knapsack / ILP) and join-order search (DP, greedy).

Dataset (documented synthetic, seed 0): customer 60,000; orders 600,000; lineitem 2,400,000; part 60,000; supplier 500
(about 3.1 million rows). DBMS: SQLite (python sqlite3), default settings.
"""
from __future__ import annotations

import itertools
import os
import sqlite3
import time

import numpy as np
from ortools.linear_solver import pywraplp

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "results", "build", "h.db")
SIZES = {"customer": 60_000, "orders": 600_000, "lineitem": 2_400_000, "part": 60_000, "supplier": 500}

SCHEMA = """
CREATE TABLE customer(c_id INTEGER PRIMARY KEY, c_nation INT, c_segment INT, c_balance REAL);
CREATE TABLE orders(o_id INTEGER PRIMARY KEY, o_cust INT, o_date INT, o_status INT, o_total REAL);
CREATE TABLE lineitem(l_id INTEGER PRIMARY KEY, l_order INT, l_part INT, l_supp INT, l_qty INT, l_price REAL, l_ship INT, l_flag INT);
CREATE TABLE part(p_id INTEGER PRIMARY KEY, p_type INT, p_size INT);
CREATE TABLE supplier(s_id INTEGER PRIMARY KEY, s_nation INT);
"""


def build_db(path=DB, seed=0):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    n = SIZES
    con.executemany("INSERT INTO customer VALUES(?,?,?,?)", zip(range(1, n["customer"] + 1), rng.integers(0, 25, n["customer"]).tolist(),
                    rng.integers(0, 5, n["customer"]).tolist(), rng.random(n["customer"]).tolist()))
    con.executemany("INSERT INTO orders VALUES(?,?,?,?,?)", zip(range(1, n["orders"] + 1), rng.integers(1, n["customer"] + 1, n["orders"]).tolist(),
                    rng.integers(0, 2500, n["orders"]).tolist(), rng.integers(0, 3, n["orders"]).tolist(), (rng.random(n["orders"]) * 1000).tolist()))
    con.executemany("INSERT INTO lineitem VALUES(?,?,?,?,?,?,?,?)", zip(range(1, n["lineitem"] + 1), rng.integers(1, n["orders"] + 1, n["lineitem"]).tolist(),
                    rng.integers(1, n["part"] + 1, n["lineitem"]).tolist(), rng.integers(1, n["supplier"] + 1, n["lineitem"]).tolist(),
                    rng.integers(1, 50, n["lineitem"]).tolist(), (rng.random(n["lineitem"]) * 100).tolist(),
                    rng.integers(0, 2500, n["lineitem"]).tolist(), rng.integers(0, 4, n["lineitem"]).tolist()))
    con.executemany("INSERT INTO part VALUES(?,?,?)", zip(range(1, n["part"] + 1), rng.integers(0, 150, n["part"]).tolist(), rng.integers(1, 50, n["part"]).tolist()))
    con.executemany("INSERT INTO supplier VALUES(?,?)", zip(range(1, n["supplier"] + 1), rng.integers(0, 25, n["supplier"]).tolist()))
    con.commit()
    con.execute("ANALYZE")
    con.close()


# name -> (weight, SQL)
WORKLOAD = {
    "Q1_ship_range": (1, "SELECT COUNT(*), SUM(l_price) FROM lineitem WHERE l_flag=1 AND l_ship BETWEEN 100 AND 160"),
    "Q2_nation_orders": (1, "SELECT SUM(o_total) FROM orders WHERE o_cust IN (SELECT c_id FROM customer WHERE c_nation=5 AND c_segment=2)"),
    "Q3_orders_lines": (1, "SELECT SUM(l_price) FROM orders JOIN lineitem ON l_order=o_id WHERE o_date BETWEEN 500 AND 520"),
    "Q4_part_lines": (1, "SELECT COUNT(*) FROM lineitem JOIN part ON l_part=p_id WHERE p_type=7"),
    "Q5_four_way": (1, "SELECT COUNT(*), SUM(l_price) FROM customer JOIN orders ON o_cust=c_id JOIN lineitem ON l_order=o_id JOIN part ON l_part=p_id "
                       "WHERE c_nation=3 AND c_segment=1 AND o_status=0 AND p_type=10"),
    "Q6_group_supplier": (1, "SELECT l_supp, SUM(l_qty) FROM lineitem WHERE l_flag=2 GROUP BY l_supp ORDER BY l_supp"),
    "Q7_status_date": (1, "SELECT COUNT(*) FROM orders WHERE o_status=1 AND o_date<100"),
    "Q8_supplier_nation": (1, "SELECT COUNT(*), SUM(l_qty) FROM supplier JOIN lineitem ON l_supp=s_id WHERE s_nation=4"),
}

CANDIDATES = {
    "li_ship": "lineitem(l_ship)", "li_flag_ship": "lineitem(l_flag,l_ship)", "li_order": "lineitem(l_order)", "li_part": "lineitem(l_part)",
    "li_supp": "lineitem(l_supp)", "li_flag_supp_qty": "lineitem(l_flag,l_supp,l_qty)", "li_order_price": "lineitem(l_order,l_price)",
    "o_cust": "orders(o_cust)", "o_date": "orders(o_date)", "o_status_date": "orders(o_status,o_date)", "o_date_total": "orders(o_date,o_total)",
    "c_nation_seg": "customer(c_nation,c_segment)", "c_nation": "customer(c_nation)", "p_type": "part(p_type)", "s_nation": "supplier(s_nation)",
}


def connect(path=DB):
    con = sqlite3.connect(path)
    con.execute("PRAGMA cache_size=-200000")
    return con


def drop_all_indexes(con):
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'ix_%'").fetchall():
        con.execute(f"DROP INDEX {name}")
    con.commit()


def create_index(con, key):
    con.execute(f"CREATE INDEX ix_{key} ON {CANDIDATES[key]}")
    con.commit()


def db_pages(con):
    return con.execute("PRAGMA page_count").fetchone()[0] * con.execute("PRAGMA page_size").fetchone()[0]


def run_query(con, sql, reps=3, timeout_s=None):
    """Return (best seconds, result rows). Rows are sorted for comparison. None time = timeout."""
    best, rows = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        if timeout_s:
            con.set_progress_handler(lambda: 1 if time.perf_counter() - t0 > timeout_s else 0, 100_000)
        try:
            rows = sorted(map(tuple, con.execute(sql).fetchall()))
        except sqlite3.OperationalError:
            con.set_progress_handler(None, 0)
            return None, None
        con.set_progress_handler(None, 0)
        best = min(best, time.perf_counter() - t0)
    return best, rows


QUERY_TIMEOUT_S = 30.0


def workload_latency(con, reps=3):
    """Per-query best latency. A query that exceeds the timeout is recorded at the timeout value (censored) with result None."""
    lat, res = {}, {}
    for q, (w, sql) in WORKLOAD.items():
        t, rows = run_query(con, sql, reps, timeout_s=QUERY_TIMEOUT_S)
        lat[q], res[q] = (QUERY_TIMEOUT_S if t is None else t), rows
    return lat, res


def weighted(lat):
    return sum(WORKLOAD[q][0] * t for q, t in lat.items())


# ------------------------------------------------------------- index selection
def select_greedy(benefit_total, size, budget):
    """Additive model: benefit/size ratio greedy."""
    chosen, used = [], 0.0
    for k in sorted(benefit_total, key=lambda k: -benefit_total[k] / size[k]):
        if benefit_total[k] > 0 and used + size[k] <= budget:
            chosen.append(k); used += size[k]
    return chosen


def select_dp(benefit_total, size, budget, unit=0.1):
    """Additive model: exact 0/1 knapsack by dynamic programming over sizes rounded up to `unit` (MB)."""
    keys = list(benefit_total)
    w = [int(np.ceil(size[k] / unit)) for k in keys]
    cap = int(budget / unit)
    T = np.zeros((len(keys) + 1, cap + 1))
    for i, k in enumerate(keys, 1):
        T[i] = T[i - 1]
        if w[i - 1] <= cap and benefit_total[k] > 0:
            T[i, w[i - 1]:] = np.maximum(T[i - 1, w[i - 1]:], T[i - 1, :cap + 1 - w[i - 1]] + benefit_total[k])
    chosen, c = [], cap
    for i in range(len(keys), 0, -1):
        if T[i, c] != T[i - 1, c]:
            chosen.append(keys[i - 1]); c -= w[i - 1]
    return chosen


def select_ilp(benefit_q, size, budget):
    """Interaction-aware ILP: each query uses at most one of the chosen indexes (its best); maximise total saving.
    benefit_q[q][k] = latency saving of query q if only index k exists."""
    s = pywraplp.Solver.CreateSolver("SCIP")
    keys = list(size)
    x = {k: s.BoolVar(f"x_{k}") for k in keys}
    y = {(q, k): s.BoolVar("") for q in benefit_q for k in keys if benefit_q[q][k] > 0}
    for (q, k), v in y.items():
        s.Add(v <= x[k])
    for q in benefit_q:
        s.Add(sum(v for (qq, k), v in y.items() if qq == q) <= 1)
    s.Add(sum(size[k] * x[k] for k in keys) <= budget)
    s.Maximize(sum(benefit_q[q][k] * v for (q, k), v in y.items()))
    s.Solve()
    return [k for k in keys if x[k].solution_value() > 0.5]


# ----------------------------------------------------------------- join order
TABLES = ["customer", "orders", "lineitem", "part"]
EDGES = {("customer", "orders"): "orders.o_cust=customer.c_id", ("orders", "lineitem"): "lineitem.l_order=orders.o_id",
         ("lineitem", "part"): "lineitem.l_part=part.p_id"}
FILTERS = {"customer": "c_nation=3 AND c_segment=1", "orders": "o_status=0", "lineitem": "1=1", "part": "p_type=10"}
PK_SIZE = {"customer": SIZES["customer"], "orders": SIZES["orders"], "part": SIZES["part"]}   # join selectivity = 1/|PK table|


def edge_between(a, b):
    return (a, b) in EDGES or (b, a) in EDGES


def forced_sql(order):
    """Join order forced with CROSS JOIN (SQLite never reorders tables across CROSS JOIN)."""
    sql = "SELECT COUNT(*), SUM(l_price) FROM " + " CROSS JOIN ".join(order) + " WHERE "
    conds = [f"({FILTERS[t]})" for t in TABLES if FILTERS[t] != "1=1"] + [c for c in EDGES.values()]
    return sql + " AND ".join(conds)


def filtered_cards(con):
    return {t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE {FILTERS[t]}").fetchone()[0] for t in TABLES}


def est_card(subset, base):
    """Independence-assumption cardinality of joining `subset`: product of filtered sizes times 1/|PK| per join edge."""
    c = 1.0
    for t in subset:
        c *= base[t]
    # each edge is FK -> PK: customer.c_id, orders.o_id, part.p_id, so its selectivity is 1/|PK table|
    pk_of_edge = {("customer", "orders"): "customer", ("orders", "lineitem"): "orders", ("lineitem", "part"): "part"}
    for (a, b), pk in pk_of_edge.items():
        if a in subset and b in subset:
            c /= PK_SIZE[pk]
    return c


def order_cost(order, base):
    """C_out: sum of estimated intermediate result sizes along a left-deep order (cross products allowed but costly)."""
    return sum(est_card(order[:i], base) for i in range(2, len(order) + 1)) + base[order[0]]


def join_order_dp(base):
    """Selinger-style dynamic programming over subsets (left-deep, avoiding cross products)."""
    best = {frozenset([t]): (base[t], [t]) for t in TABLES}
    for size in range(2, len(TABLES) + 1):
        for sub in itertools.combinations(TABLES, size):
            S = frozenset(sub)
            cand = None
            for t in sub:
                rest = S - {t}
                if rest in best and any(edge_between(t, r) for r in rest):
                    cost = best[rest][0] + est_card(sub, base)
                    if cand is None or cost < cand[0]:
                        cand = (cost, best[rest][1] + [t])
            if cand:
                best[S] = cand
    return best[frozenset(TABLES)][1]


def join_order_greedy(base):
    """Start from the smallest filtered table, then keep adding the connected table giving the smallest intermediate result."""
    order = [min(TABLES, key=lambda t: base[t])]
    while len(order) < len(TABLES):
        cands = [t for t in TABLES if t not in order and any(edge_between(t, r) for r in order)]
        order.append(min(cands, key=lambda t: est_card(order + [t], base)))
    return order
