# Track J - Software-Engineering Optimization: Test-Suite Minimization and Prioritization

**Codebase:** the public open-source package **networkx 3.7**, subpackage `networkx/algorithms/shortest_paths` (4,595 lines of source in its non-test files; 654 tests). Domain: software engineering, test-suite minimization and prioritization.
Pipeline: `python collect.py networkx/algorithms/shortest_paths` (per-test line coverage with coverage.py contexts + per-test runtimes) -> `python mutate.py networkx/algorithms/shortest_paths 48` (mutation testing) -> `python optimize.py` -> `python wallclock.py`.

## Profiling and bottleneck
The full suite takes 8.2 s end to end (median of 3): 4.9 s of test execution reported by pytest plus about 3 s of start-up, import and collection. The summed per-test runtimes are only 2.9 s, so the suite is dominated by fixed overhead and a long tail of tiny tests. 654 tests cover 800 source lines.

## Formulation
**Minimization = weighted set cover.** Variables x_t in {0,1} (keep test t). Minimise sum_t time_t x_t subject to: every line covered by the full suite is covered by at least one kept test. Also solved with unit cost (fewest tests).
Methods: naive baselines (keep everything; random order until covered; a random subset of the same size), **greedy** (most newly covered lines per second; the classic ln(n) approximation for set cover), **ILP** (exact, SCIP).
**Cost in correctness** is measured by mutation testing: 48 operator-swap mutants on covered lines (seed 0), each run against the full suite in an isolated copy of networkx; 33 were killed by the full suite (the other 15 are equivalent or simply untested and are excluded). A reduced suite's *mutation score* = fraction of those 33 it still kills.
**Prioritization:** order tests so faults are found early; metric APFD (average percentage of faults detected).

## Results (`results/summary.json`, `results/wallclock.json`)

| Suite | tests | summed test time | line coverage | mutants still killed |
|---|---|---|---|---|
| full suite (baseline) | 654 | 2.90 s | 100% | 33 / 33 |
| **ILP, time-weighted (proven optimal)** | **61** | **0.090 s** | 100% | 30 / 33 (90.9%) |
| greedy, time-weighted | 67 | 0.096 s | 100% | 29 / 33 (87.9%) |
| ILP, fewest tests (proven optimal) | 60 | 0.094 s | 100% | 29 / 33 |
| greedy, fewest tests | 60 | 0.107 s | 100% | 29 / 33 |
| random order until covered (mean of 30) | 77.2 | 0.136 s | 100% | 30.3 / 33 (91.7%) |
| random 61 tests (mean of 200) | 61 | 0.271 s | 51.4% | 12.2 / 33 (37.1%) |

**Wall-clock (3 runs each, including pytest start-up):**

| How the reduced suite is run | wall time | speedup vs full (8.2 s) |
|---|---|---|
| naive: 61 node ids on the command line | 15.9 s | **0.51x (slower)** |
| collect once, filter with a small plugin (`select_plugin.py`) | 4.7 s | **1.74x** |

Findings:
- **The optimization is real but much smaller than the model says.** The ILP cuts summed test time 32x (2.90 -> 0.09 s), test execution 3.3x (4.92 -> 1.47 s), but end to end only 1.74x, because about 3 s of start-up and collection does not shrink. Optimizing the modeled objective (per-test time) is not the same as optimizing the measured one.
- **The obvious implementation made it slower.** Passing the selected node ids to pytest re-collects a file once per id, so the 61-test run took 15.9 s, twice the full suite. The plugin that collects once and deselects fixed this. This was found only by measuring.
- **The cost in correctness is real.** Keeping 100% line coverage still loses 3 of 33 detectable faults (ILP), or 4 (greedy): coverage is a proxy for fault detection, not the same thing. Random covering sets (77 tests) kept as many faults (91.7%) as the ILP set, at 50% more test time, so the optimizer buys speed, not better detection. With 33 mutants, a difference of one mutant (about 3 points) is within noise.
- **Coverage-guided selection matters:** a random subset of the same size keeps only 51% coverage and 37% of the faults.
- **Greedy vs exact:** greedy was within 7% of the optimal time and 10% more tests, and needed no solver; both ILPs solved to proven optimality in well under a second at this size.
- **Prioritization:** APFD 0.962 for greedy additional-coverage-per-second ordering vs 0.743 for random orders and 0.304 for the original alphabetical order (the fault-revealing tests happen to sort late). The greedy order uses only coverage, never the mutant results, so there is no leakage.

## Maintainability and readability cost
A minimised suite must be recomputed whenever code or tests change (coverage collection here takes about 20 s), it hides 593 tests from the default run, and 3 of 33 faults would slip through. Sensible use: run the minimised, prioritised suite on every commit for fast feedback, and the full suite before merging.

## Limitations
One subpackage of one project; line coverage only (no branch coverage); 48 simple operator mutants (no statement deletion etc.); runtimes from one machine; mutants surviving the full suite were not inspected for equivalence.
