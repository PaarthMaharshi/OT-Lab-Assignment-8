# Prompt log entry - Track E simulator

- **Track / task:** E - tower-defense simulator, surrogate objective, random baseline
- **Date:** 2026-11-06
- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`)

## 1. Original prompt
"Track E simulator" (after choosing tower defense as the main case study).

## 2. AI response (relevant excerpt)
Generated `td/sim.py`, `maps.py`, `surrogate.py`, `baselines.py`, `viz.py`, tests, `run_baseline.py`.

## 3. Revised prompt
None needed; follow-ups were caused by AI mistakes (see section 6).

## 4. Resulting code change
Commit: [6ab8663](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/6ab8663). Files: `tracks/E_tower_defense/**`.

## 5. Verification note
- 12 pytest tests pass. Expected values were computed by hand BEFORE running (tower in range for 3 ticks = 30 damage; kill at hp 20; slow-tower case = 5 shots = 50 damage).
- Surrogate checked for monotonicity and diminishing returns on 200 random set pairs.
- `run_baseline.py` seeds 0..99, deterministic.

## 6. Issues found (feeds docs/failure_log.md)
1. AI left a syntactically invalid placeholder line in the test file; caught when reviewing the file, removed before running.
2. Default budget 120 made the benchmark trivial (all 100 random layouts killed every enemy, std 0), so no method could show improvement. Found by reading the output; budget lowered to 30 after a sweep (20/30/40/60).
3. Surrogate overestimates the true simulator score (budget 30: 2484 vs 1757 mean). It ignores slow towers, single-target contention and spawn overlap. Must be reported, not hidden, when comparing ILP/greedy results.

## 7. Chat export / screenshot
_add export_
