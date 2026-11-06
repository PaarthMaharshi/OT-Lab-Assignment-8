import itertools
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hlib as h  # noqa: E402


def brute_additive(benefit, size, budget):
    keys = list(benefit)
    best = 0
    for m in range(2 ** len(keys)):
        ch = [k for i, k in enumerate(keys) if m >> i & 1]
        if sum(size[k] for k in ch) <= budget:
            best = max(best, sum(benefit[k] for k in ch))
    return best


def brute_interaction(bq, size, budget):
    keys = list(size)
    best = 0
    for m in range(2 ** len(keys)):
        ch = [k for i, k in enumerate(keys) if m >> i & 1]
        if sum(size[k] for k in ch) <= budget:
            best = max(best, sum(max([bq[q][k] for k in ch] + [0]) for q in bq))
    return best


def test_knapsack_dp_matches_brute_force_and_greedy_is_feasible():
    rng = np.random.default_rng(0)
    for _ in range(20):
        keys = [f"k{i}" for i in range(8)]
        size = {k: float(rng.integers(1, 10)) for k in keys}          # integer MB so unit rounding is exact
        ben = {k: float(rng.random() * 10) for k in keys}
        B = 15.0
        dp = h.select_dp(ben, size, B, unit=1.0)
        assert sum(size[k] for k in dp) <= B
        assert sum(ben[k] for k in dp) == pytest.approx(brute_additive(ben, size, B))
        g = h.select_greedy(ben, size, B)
        assert sum(size[k] for k in g) <= B and sum(ben[k] for k in g) <= brute_additive(ben, size, B) + 1e-9


def test_interaction_ilp_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(15):
        keys = [f"k{i}" for i in range(7)]
        size = {k: float(rng.integers(1, 10)) for k in keys}
        bq = {f"q{j}": {k: float(rng.random() * 5 * (rng.random() < 0.5)) for k in keys} for j in range(5)}
        B = 14.0
        ch = h.select_ilp(bq, size, B)
        got = sum(max([bq[q][k] for k in ch] + [0]) for q in bq)
        assert sum(size[k] for k in ch) <= B and got == pytest.approx(brute_interaction(bq, size, B))


def test_join_order_dp_equals_brute_force_over_connected_orders():
    rng = np.random.default_rng(2)
    for _ in range(30):
        base = {t: float(rng.integers(1, 100000)) for t in h.TABLES}
        valid = [o for o in itertools.permutations(h.TABLES)
                 if all(any(h.edge_between(o[i], o[j]) for j in range(i)) for i in range(1, len(o)))]
        best = min(h.order_cost(list(o), base) for o in valid)
        assert h.order_cost(h.join_order_dp(base), base) == pytest.approx(best)
        g = h.join_order_greedy(base)
        assert h.order_cost(g, base) >= best - 1e-6


def test_forced_join_orders_return_identical_results_on_a_tiny_database(tmp_path):
    con = sqlite3.connect(":memory:")
    con.executescript(h.SCHEMA)
    rng = np.random.default_rng(3)
    con.executemany("INSERT INTO customer VALUES(?,?,?,?)", [(i, 3, 1, 0.5) for i in range(1, 30)])
    con.executemany("INSERT INTO orders VALUES(?,?,?,?,?)", [(i, int(rng.integers(1, 30)), 5, 0, 1.0) for i in range(1, 200)])
    con.executemany("INSERT INTO lineitem VALUES(?,?,?,?,?,?,?,?)", [(i, int(rng.integers(1, 200)), int(rng.integers(1, 40)), 1, 2, float(i % 7), 3, 1) for i in range(1, 800)])
    con.executemany("INSERT INTO part VALUES(?,?,?)", [(i, 10 if i % 3 == 0 else 1, 4) for i in range(1, 40)])
    results = set()
    for order in itertools.permutations(h.TABLES):
        results.add(tuple(con.execute(h.forced_sql(order)).fetchall()))
    assert len(results) == 1 and results.pop()[0][0] > 0


def test_est_card_basics():
    base = {"customer": 10.0, "orders": 100.0, "lineitem": 1000.0, "part": 5.0}
    assert h.est_card(["customer"], base) == 10
    assert h.est_card(["customer", "orders"], base) == pytest.approx(10 * 100 / h.PK_SIZE["customer"])
