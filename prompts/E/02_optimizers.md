# Prompt log entry - Track E optimizers (greedy, ILP, annealing)

- **Track / task:** E - placement optimizers and fair comparison
- **Date:** 2026-11-06
- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`)

## 1. Original prompt
"Yes, build them" (greedy, ILP and annealing after the simulator was done).

## 2. AI response (relevant excerpt)
`td/optim.py` (greedy, ILP via OR-Tools SCIP, simulated annealing), `tests/test_optim.py`, `compare.py`.

## 3. Revised prompt
None typed by me; the AI revised its own surrogate after reading the first results (section 6).

## 4. Resulting code change
Commit: _not committed yet_. Files: `td/optim.py`, `td/surrogate.py`, `compare.py`, tests.

## 5. Verification note
- ILP checked against brute-force enumeration on a tiny instance (budgets 10/30/40): equal.
- Greedy valid, never above the ILP optimum on the surrogate, and above 0.316 x optimum (the knapsack bound; NOT 1-1/e).
- Annealing deterministic for a fixed seed, valid, never worse than its start.
- Every method scored by the same true simulator; SA seeds 0-4, fixed 800 simulator evaluations.

## 6. Issues found (feed docs/failure_log.md)
1. **Misleading surrogate (v1).** The first surrogate let every tower damage every passing enemy independently. One gun scored "2860/2860" on the surrogate but only 915 in the simulator. Greedy (915) did WORSE than random (mean 1757), and ILP only looked good (2520) by luck because any layout reaching 2860 was "optimal". Detected from the comparison table (greedy below the random mean, identical results at two budgets). Fix: add a per-tower firing capacity (one target per tick), giving a max-flow surrogate that is still submodular. Lesson: validate a surrogate against the real simulator before trusting an optimizer that uses it.
2. **Wrong guarantee in the draft.** A cardinality bound (1-1/e) would have been claimed for greedy; the budget is a knapsack constraint, so the bound used is (1/2)(1-1/e) = 0.316.
3. **Saturated instances.** Budget 40 and above lets every method kill all enemies; comparison restricted to budgets 20 and 30.

## 7. Chat export / screenshot
_add export_
