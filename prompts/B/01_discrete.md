# Prompt log entry - Track B discrete optimization

- **Track / task:** B - knapsack (greedy/DP/B&B/ILP), Max-Cut QUBO and Ising, feature selection, penalty terms, weighted interval DP
- **Date:** 2026-11-06
- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`)

## 1. Original prompt
"Start Track B" (after Tracks A and E).

## 2. AI response (relevant excerpt)
`knapsack.py`, `qubo.py`, `featsel.py`, `scheduling.py`, `tests/test_discrete.py`, `experiments.py`.

## 3. Revised prompt
None typed by me.

## 4. Resulting code change
Commit: [aeac9ab](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/aeac9ab) (discrete DL choices added in [fe6685c](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/fe6685c)). Files: `tracks/B_discrete/**`.

## 5. Verification note
- 13 tests: slide examples (feature-selection tables, knapsack B+D=6, QUBO min -4 at 0101/1010, Ising -5/2 constant, soft threshold, 955 fits), brute-force agreement for knapsack / Max-Cut / intervals, and QUBO-vs-Ising energy equality over all 2^n assignments.
- Experiments assert that exact methods agree before reporting any timing.

## 6. Issues found (feed docs/failure_log.md)
1. **Invented number.** The first experiment script hard-coded "40 model fits" for the Lasso row. It was never measured. Found on review of the results table; replaced by a real counter (28.2 on average). Lesson: every reported count must come from instrumentation.
2. **Untuned hyperparameter used before the evidence.** The QUBO feature selection first used penalty P=1.0, but the penalty sweep showed large P degrades solutions. Reran with P=0.3 chosen from the sweep (training-data objective only).
3. **Redundant code.** `qubo_to_ising` contained three overwritten assignments for J while the AI worked out the coefficient; cleaned with an explanatory comment. A derivation left in code is a readability bug, not harmless.
4. **Misleading reference at n >= 50.** "Best known" for large Max-Cut graphs includes the heuristics' own results, so they cannot appear worse than it; documented as a caveat rather than hidden.
5. **Sloppy test setup.** `test_cardinality_penalty_expansion` generated the same random data twice. Works, but is confusing and should be tidied before the viva.

## 7. Chat export / screenshot
_add export_
