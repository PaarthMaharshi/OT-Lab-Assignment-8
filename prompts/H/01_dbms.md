# Prompt log entry - Track H

- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`), 2026-11-07
- **Prompt:** "just do everything else without asking me" / "do it quick please" (Track H from the assignment PDF)
- **Output:** `hlib.py`, `experiments.py`, 5 tests. Commit: [5826744](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/5826744).
- **Verification:** knapsack DP and the interaction ILP equal brute force; join-order DP equals brute force on the cost model; result sets of every plan and forced join order are identical to the baseline.

## Issues found (feed docs/failure_log.md)
1. **Wrong index sizes.** Sizes were measured as page-count deltas on one database, but dropped indexes leave free pages that later indexes reuse, so 11 of 15 candidates were measured as 0 MB. Detected from the printed sizes; fixed by building every candidate on a fresh copy of the baseline file.
2. **Benchmark too light.** The first dataset (1M rows) gave a 0.47 s workload with per-index savings of 0.02-0.15 s, about the same size as timing noise. Scaled to 3.1M rows.
3. **Hang with no timeout.** One index set caused a pathological join plan and the experiment ran for minutes with no output. Added a 30 s per-query timeout (via a progress handler), recorded as censored.
4. **Dead code** in `est_card` (leftover primary-key mapping lines) found on review and removed before running.
5. **Surprising result kept:** index sets chosen by the additive models made the workload 11x slower at one budget (planner regression); I diagnosed it (Q5 plan) rather than discarding the run.
6. **Shell timing quirk:** sleeping inside shell wait loops returned early in this environment, so long runs were polled through their log instead.
