# Track H - DBMS Optimization (SQLite)

**DBMS:** SQLite 3 via Python `sqlite3`, default planner. **Data:** documented synthetic database (seed 0): customer 60,000, orders 600,000, lineitem 2,400,000, part 60,000, supplier 500 (about 3.1 million rows, 98 MB, no indexes). **Workload:** 8 queries (selective scans, a grouped scan, 2-, 3- and 4-way joins).
Run: `python experiments.py` (long: about 20 minutes, mostly index builds and timeouts). Tests: `python -m pytest tests` (5 tests: DP knapsack and the interaction ILP equal brute force; DP join order equals brute force on the cost model; all 24 forced join orders return identical results).
Every plan's result set is compared with the baseline result set (asserted identical; a timed-out query is excluded from the comparison).

## Formulations

**Index selection.** 15 candidate indexes (single and composite). Variables x_k in {0,1}; storage budget B (measured index size). Two models:
- *additive* (knapsack): maximise sum_k b_k x_k s.t. sum_k s_k x_k <= B, where b_k = workload time saved by index k alone. Solved by **greedy** (benefit/size) and **dynamic programming**.
- *interaction-aware ILP*: each query uses at most its best chosen index: maximise sum_q sum_k b_qk y_qk, y_qk <= x_k, sum_k y_qk <= 1, sum_k s_k x_k <= B. Solved exactly with SCIP.
Each chosen set is then **built on a fresh copy of the database, `ANALYZE`d and timed on the real workload**; predictions are compared with the measurement.

**Join order** (4-way query: customer, orders, lineitem, part). Forced with `CROSS JOIN` (SQLite never reorders across it); all 4! = 24 orders timed (20 s timeout), with all indexes present. A Selinger-style **DP over subsets** (left-deep, no cross products) and a **greedy** (smallest filtered table first, then the connected table giving the smallest intermediate result) both minimise estimated C_out (sum of intermediate result sizes, independence assumption); they are compared with SQLite's own planner and the measured best.

## Results

**Index selection** (baseline workload 1.53 s; all 15 indexes: 0.134 s, 258 MB; actual saving in seconds of weighted workload time)

| Budget | greedy (additive) | DP (additive) | ILP (interaction-aware) |
|---|---|---|---|
| 15% (39 MB) | +0.175 | +0.173 | +0.155 (same 4 indexes as greedy) |
| 30% (77 MB) | **-15.9** | **-15.8** | **+0.564** |
| 60% (155 MB) | +1.334 | +0.706 | +1.362 |

Findings:
- **More indexes can make the workload much slower.** At the 30% budget the greedy and DP sets (which lack a `part(p_type)` index but contain `li_part`) pushed Q5 from 0.2 s to about 28 s: SQLite built an automatic partial index on `part` and joined customer x part before touching lineitem, producing a large intermediate result. Q1 also regressed (0.15 -> 1.2 s). The ILP set at the same budget avoided this. The ILP's model does not know about plan regressions, so avoiding it is partly luck (it picked a different set), not proof that the ILP is safe.
- **Benefits are not additive.** For the identical 4-index set at 15% the additive model predicted 0.79 s, the interaction-aware model 0.69 s, and the measured saving was 0.16-0.18 s: each benefit was measured with one index alone, but the planner's choices change when indexes are combined. Both models over-predict; the real objective needs measuring.
- At 15% all three methods chose the same indexes (differences of 0.02 s are measurement noise). At 60% DP's set (0.83 s) is clearly worse than greedy (0.20 s) and ILP (0.17 s), because it rounds sizes up to 0.1 MB units and optimises the additive model.
- Storage matters: the best 60%-budget set uses 155 MB, more than the 98 MB database.

**Join order** (Q5, indexes present; SQLite's own planner: 0.037 s)

| Method | order | time |
|---|---|---|
| measured best of 24 | customer -> orders -> lineitem -> part | 0.041 s |
| **DP** (estimated C_out) | customer -> orders -> lineitem -> part | **0.041 s** (equals the measured best) |
| greedy | part -> lineitem -> orders -> customer | 0.049 s (2nd best) |
| worst completed | customer -> part -> lineitem -> orders | 15.6 s |
| timed out (> 20 s) | 7 of 24 orders | - |

The best order is 380x faster than the worst completed order, so ordering matters enormously. Over the 17 completed orders the Spearman rank correlation between estimated cost and measured time is 0.57 (p = 0.018): the model identifies the right order here but is only a moderate predictor overall. SQLite's default planner matched the best order, so the DP's contribution is reproducing the engine's choice, not beating it.

`EXPLAIN QUERY PLAN` before/after are stored in `results/summary.json`. With all indexes, Q5 runs part (`ix_p_type`) -> lineitem (`ix_li_part`) -> orders (primary key) -> customer (covering `ix_c_nation_seg`) and Q3 runs orders (`ix_o_date`) -> lineitem (covering `ix_li_order_price`); without indexes SQLite scans lineitem or builds automatic indexes.

## Limitations
- One synthetic dataset and one 8-query workload; uniformly random data, so the independence assumption in the cost model is nearly true. Real skew would hurt the estimates.
- SQLite only; results about planner regressions are specific to its planner. No materialised views or partitioning.
- Timings are single-machine best-of-3 (join-order runs are single measurements); differences below about 0.05 s of total workload time are noise.
- The additive models were fed benefits measured on index-free copies; a one-at-a-time greedy that re-measures after each pick would do better but costs many more builds.
