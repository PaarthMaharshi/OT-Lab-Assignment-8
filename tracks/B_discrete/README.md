# Track B - Discrete Optimization

Run: `python experiments.py` (about 6 minutes, most of it the exhaustive feature search); rerun one part with `python experiments.py exp_featsel`. Tests: `python -m pytest tests` (13 tests, expected values from the lecture slides and from brute-force enumeration).

| Requirement | Code | Evidence |
|---|---|---|
| Greedy | `knapsack.greedy_ratio`, `featsel.greedy_forward`, `backward_elimination` | slide 24/25 examples reproduced exactly (forward picks {A,B}=0.80, optimum {C,D}=0.84, backward finds {C,D}) |
| ILP | `knapsack.ilp`, `qubo.ilp_maxcut`, `scheduling.ilp` (OR-Tools SCIP) | agrees with DP / brute force on every instance where it proves optimality |
| Branch-and-Bound | `knapsack.branch_and_bound` (LP-relaxation bound, own implementation) | node counts below; slide 26 knapsack (B+D = 6) reproduced |
| Dynamic programming | `knapsack.dp`, `scheduling.weighted_dp` | exact, checked against ILP and brute force |
| Feature selection (combinatorial) | `featsel.py`, exp_featsel | breast-cancer data, d=30, k=4, five splits |
| QUBO / Ising with penalty terms | `qubo.py`, `featsel.feature_qubo`, exp_qubo_penalty | energy equality for all 2^n assignments (test); slide 28 values (min -4, H = 1/2 sum ZZ - 5/2) |
| L1 sparsity | Lasso in exp_featsel; soft-threshold test (slide 27) | |
| Baseline for Track C | `results/maxcut_instances.json` | 12 small Max-Cut graphs (6-12 nodes) with brute-force optimum |

## Findings (`results/summary.json`; figures `results/fig1_knapsack.png`, `fig2_maxcut.png`)

**Knapsack (3 instances per cell; exact methods always agreed).** Greedy is nearly free (0.2-0.4 ms) and within 0.1-0.4% of optimal on uncorrelated items, but up to 9.8% off (n=20) on strongly correlated ones. B&B needs only 55-633 nodes on uncorrelated items, yet on strongly correlated items at n=80 it hit the 150,000-node cap on all 3 instances without proving optimality; at n=160 it proved 2 of 3, so difficulty is instance-dependent, not monotone in n. DP wins here (0.2-7 ms) only because weights are small integers (capacity about 1,000-4,000); its cost grows with capacity, which B&B and ILP do not depend on. ILP costs 12-22 ms, mostly solver overhead.

**Max-Cut on G(n, 0.5), 3 graphs per size.** Brute force is exact but doubles per node (2.9 s at n=22); ILP proves optimality up to n=30 (4.6 s), then times out at 20 s for n=50 and 100 (0 of 3 proven) returning cuts at 97.5% and 90.2% of the best known. Simulated annealing (10 seeds) reaches 99.9-100% of best known in 0.05-0.17 s; 1-flip local search with restarts reaches 99.4-99.5%. Caveat: for n >= 50 "best known" is the best any method found, so the heuristics cannot look suboptimal against it.

**Feature selection (k=4 of 30, test accuracy mean +- std over 5 splits, 114 test samples).**

| Method | model fits | test acc |
|---|---|---|
| forward greedy | 114 | 0.961 +- 0.007 |
| backward elimination | 455 | 0.968 +- 0.009 |
| Lasso (L1 path) | 28 | 0.967 +- 0.007 |
| QUBO + SA (relevance - redundancy, cardinality penalty) | 0 | 0.951 +- 0.013 |
| random subset | 0 | 0.919 +- 0.011 |
| exhaustive (split 0 only) | 27,405 (122 s) | 0.965 |

Everything except random is within noise of the exhaustive optimum (one test sample is 0.9 points), so the honest conclusion is: greedy, backward and Lasso lose nothing measurable here while using 240x to 1000x fewer fits. The QUBO route needs no model fits but is slightly worse because it optimizes a proxy (correlations), not accuracy. The exhaustive winner is selected on a validation set, so its 1.0 validation score is optimistic.

**Penalty terms (k=5, 20 SA runs per penalty).** The exact feasible optimum of the objective is -1.255. With P <= 0.05 the cardinality constraint is ignored (0/20 runs returned 5 features; SA returned 4); P=0.1 gives 10/20; P >= 0.2 always feasible. But solution quality then degrades as P grows: the mean gap to the exact optimum rises from 0.027 (P=0.2) to 0.197 (P=2), because a large penalty swamps the objective's small differences. A usable P is a narrow window, not "as large as possible". The main feature experiment uses P=0.3, picked from this sweep (it uses only training-data objectives, no test labels).

**Weighted interval scheduling.** DP equals ILP and brute force. Greedy earliest-finish (optimal for unweighted intervals) reaches only 75-84% of the weighted optimum; greedy-by-weight 47-100%.

## Limitations

- Only Max-Cut and feature selection are written as QUBOs; scheduling or portfolio QUBOs are not done (the Track E tower-defense QUBO is still open).
- Discrete choices in deep learning (architecture search, quantization, pruning, slide 28) are not experimented on, only listed in COVERAGE.md.
- Feature-selection scores use one validation split per random split; the exhaustive search ran on split 0 only because it costs 2 minutes.
- Branch-and-bound is a plain Python implementation with a node cap, not a tuned solver; its absolute times are not comparable to SCIP.
