# Prompt log entry - Track J

- **AI tool + model version:** Claude Code. Code written with Claude Sonnet 5.5 (`claude-sonnet-5-5`); measurements and write-up finished with Claude Opus 5.5 (`claude-opus-5-5`), 2026-11-07.
- **Prompt:** "just do everything else without asking me" / "do it quick please" (Track J from the assignment PDF)
- **Output:** `collect.py`, `mutate.py`, `optimize.py`, `wallclock.py`, `select_plugin.py`. Commit: see git log.
- **Verification:** every reduced suite is checked for 100% line coverage of the full suite; ILP solutions proven optimal; fault detection measured by mutation testing; wall-clock measured end to end (3 runs).

## Issues found (feed docs/failure_log.md)
1. **Crash on an infinite-loop mutant.** No per-test timeout; one mutant hung the run past 300 s and the whole mutation run was lost. Added pytest-timeout (20 s per test), whole-run timeout handling and an unconditional restore of the mutated file.
2. **Invalid speedup (5.2x).** The first wall-clock script passed node ids relative to the wrong folder, so the "reduced" run executed no tests ("no tests ran") yet was timed. Detected by reading pytest's last output line.
3. **Optimized suite slower than the full suite.** Once fixed, naming 61 tests on the command line took 15.9 s vs 8.2 s for all 654, because pytest re-collects per id; solved with a collect-once plugin (1.74x).
4. **Leftover junk in `mutate.py`** (a no-op `if False` argument expression and unused variables) found on review before running.
5. **Escape mangling** of backslash-n in shell heredocs broke a string literal; fixed with the editor tool.
