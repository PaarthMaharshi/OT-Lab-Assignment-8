# Prompt log entry - Track E: QUBO formulation, second map, parallel evaluation

- **Track / task:** E - tower placement as a QUBO (bridge to Tracks B and C), spiral map, parallel annealing evaluation
- **Date:** 2026-11-07
- **AI tool + model version:** Claude Code; code written with Claude Sonnet 5.5, finished with Claude Opus 5.5 (`claude-opus-5-5`)

## 1. Original prompt
"just do everything else without asking me" (the Track E README listed QUBO, second map and parallel evaluation as open).

## 2. AI response (relevant excerpt)
`td/qubo_tower.py`, `td/maps.spiral_map`, `run_qubo.py`, `parallel_eval.py`, two-map `compare.py`, `tests/test_extra.py`.

## 4. Resulting code change
Commit: see git log.

## 5. Verification note
- QUBO energy equals a direct evaluation of the formula; with a large penalty the minimum has exactly k towers (test).
- Spiral path: consecutive cells adjacent, no self-touching (test).
- Small QUBO solved exactly by enumerating 2^14 states; SA and QAOA compared against it.

## 6. Issues found (feed docs/failure_log.md)
1. **Penalty too small, again.** The first QUBO used P = 3-4 while a single tower is worth about 100, so the lowest-energy states had too many towers: the full-map QUBO+SA placements were infeasible and QAOA put less probability on feasible states than uniform sampling (0.4-1.7% vs 2.2%). Fixed with P = 1.2 x the largest tower value and energies normalised for QAOA (feasible probability then 36-64%, monotone in p). This repeats the Track B lesson.
2. **JSON crash on numpy bool** discarded a finished run; converter added.
