# Prompt log entry - Track F

- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`), 2026-11-06
- **Prompt:** "do everything else without asking me" / "do it quick please" (Track F from the assignment PDF)
- **Output:** `toyc.py`, `passes.py`, `programs.py`, `experiments.py`, 12 tests. Commit: [2c7a6ce](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/2c7a6ce).
- **Verification:** VM output equals optimized output for every program, pass and random order; 150 fuzzed programs; division by zero not removed by DCE; exact ILP colouring never worse than greedy.

## Issues found (feed docs/failure_log.md)
1. **Two broken tests (mine, not the compiler's).** The strength-reduction example was constant-folded before the pass saw a multiplication; the fuzz loop iterated over tuples-of-characters and raised KeyError. Both fixed after the first test run.
2. **Timing noise reported as a speedup.** The first benchmark showed a 1.5x "speedup" for a control program whose code did not change, and a 2x for the very first executable of a program (cold start). Fixed with a warm-up run, 7 repetitions, longer workloads and an A/A noise-floor control; the report only counts gains that clearly exceed the noise.
3. **Instruction count is not run time.** Two programs cut dynamic instructions by 6% and 17% with no measurable speedup; documented as a limitation of the metric, not hidden.
4. **Shell quoting failures** when writing long files by heredoc; used the editor tool instead.
