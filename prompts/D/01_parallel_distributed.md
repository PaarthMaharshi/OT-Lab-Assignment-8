# Prompt log entry - Track D

- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`), 2026-11-06
- **Prompt:** "Start Track D", then "just do everything else without asking me" and "do it quick please"
- **Output:** `dlib.py`, `pwork.py`, `experiments.py`, 8 tests. Commit: [9cd60b6](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/9cd60b6) (supplement: [fe6685c](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/fe6685c)).
- **Verification:** slide makespans 15/9/11, Amdahl 6.4, Gustafson 14.5, bubble 3/11, MGDA vs brute force, FedAvg one round equals a centralized step, multi-task gradients vs finite differences, greedy checked against the ILP optimum.

## Issues found (feed docs/failure_log.md)
1. **Garbled formula.** The first `mgda_weights` contained a nonsense `* -1 * -1`; replaced by the closed form and tested against brute force.
2. **Shell quoting failure** when writing long files through a heredoc; the files were written with the editor tool instead.
3. **Unrepresentative benchmark.** With a wide shared layer the two tasks did not conflict, so equal weighting won and raw MGDA left the large-scale task untrained; the experiment was redesigned (one shared feature) and a normalised MGDA variant added. The raw-MGDA failure is kept in the report.
4. **Parallel result worse than hoped.** The data-parallel gradient is slower than serial at p=12 (0.83x). Reported as a result (memory-bound, logical cores), not tuned away.
