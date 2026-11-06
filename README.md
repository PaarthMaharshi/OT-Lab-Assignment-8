# Optimize Anything - 14-Day Vibe-Coding Challenge (Assignment 8)

Course: Optimization Techniques (R5CO3101T), Chapter 6. All ten tracks (A-J) are implemented, measured and tested.
Main case study (Track E): **tower-defense placement optimization**.

- **Technical report:** [REPORT.md](REPORT.md) (a section per track and answers to the 10 final questions)
- **Chapter 6 coverage checklist:** [COVERAGE.md](COVERAGE.md) (60 topics, each mapped to code and a measured or tested result)
- **AI failure log:** [docs/failure_log.md](docs/failure_log.md) (17 documented cases) - **prompt logs:** `prompts/<track>/`

## Tracks

| Track | Topic | Folder | Headline result |
|---|---|---|---|
| A | Continuous optimization | `tracks/A_continuous` | Newton / natural gradient: 4 steps vs 1938 (GD) and 3072 (Adam); Newton converges to a saddle |
| B | Discrete optimization, QUBO/Ising | `tracks/B_discrete` | B&B blows up on correlated knapsacks; SA solves Max-Cut where ILP times out; penalty weight has a narrow window |
| C | Quantum optimization (simulated) | `tracks/C_quantum_optimization` | QAOA 0.83 -> 0.91 of max cut (p=1 -> 3); barren plateau halves gradient variance per qubit |
| D | Submodular, parallel, distributed | `tracks/D_submodular_parallel_distributed` | 4.25x measured speedup; 56x from loop order; FedAvg, MGDA, GradNorm, top-k compression |
| E | **Main case study: tower defense** | `tracks/E_tower_defense` | annealing best in 11/12 cases; the surrogate model, not the algorithm, limits ILP |
| F | Compiler optimization | `tracks/F_compiler` | toy compiler, 4 passes, up to 1.51x (above an A/A noise floor); pass ordering and register allocation as search |
| G | xv6 OS optimization | `tracks/G_xv6` | lottery and MLFQ schedulers + stats syscalls; usertests pass under all policies |
| H | DBMS optimization (SQLite) | `tracks/H_dbms` | interaction-aware ILP index selection; additive selection caused an 11x regression; DP join order = measured best |
| I | Quantum algorithms and QML (simulated) | `tracks/I_quantum_algorithms_qml` | DJ/BV/Grover/QFT/QPE/Shor by hand; QML loses to classical models (honest negative result) |
| J | Software-engineering optimization | `tracks/J_software_engineering` | networkx test suite 654 -> 61 tests (ILP set cover), 1.74x end to end, 3/33 faults lost |

## Setup and reproduction

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Each track: `cd tracks/<track>` then `python -m pytest tests` (106 tests in tracks A-F, H, I) and the experiment command in its README (for example `python experiments.py`). Results are written to `tracks/<track>/results/` as JSON and PNG; seeds are fixed.
- Track F needs `gcc` (MSYS2 UCRT64 used here). Track D's `locality_sync.c` needs `gcc -fopenmp`.
- Track G needs QEMU and `riscv64-unknown-elf-gcc` (MSYS2 `clang64` packages used here; Linux works too): apply `xv6_scheduling.patch` to xv6-riscv commit `fa2ddcb`, then `bash build.sh` and `python bench.py bench`.
- Track J needs `networkx`, `coverage`, `pytest-cov`, `pytest-timeout`.

Hardware: Windows 11, 12 logical processors (most likely 6 physical cores, judging by the measured speedups). All quantum results are simulations; no real quantum hardware was used.

## AI tools register (required)

| Tool | Model / version | Tracks | Dates | Kind of work |
|---|---|---|---|---|
| Claude Code | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | setup, A, B, C, D, E, F, H, I, J (code) | 2026-11-06 to 2026-11-07 | planning, code generation, debugging, testing, documentation |
| Claude Code | Claude Opus 5.5 (`claude-opus-5-5`) | J (measurements, write-up), G, E (finish), D/B supplements, report | 2026-11-07 | code generation, debugging, measurement, documentation |
| _add every other tool you use, including for debugging or explanation_ | | | | |

## Still to do by the student (cannot be done by the AI on your behalf)

1. **Tool comparison** (`docs/tool_comparison.md`): run one task with a second AI tool (e.g. ChatGPT or Gemini) and compare it with Claude Code; then add that tool to the register above.
2. **Chat exports / screenshots** into each `prompts/<track>/` folder (the logs have a placeholder).
3. **Demonstration** (short video or live demo): suggested flow is Track E (`run_baseline.py` map, `results/comparison.png`), then one surprising result (Track H's 11x regression or Track J's slower-than-full suite).
4. **Viva preparation**: read and be able to modify `tracks/E_tower_defense/td/sim.py`, `td/optim.py`, `tracks/A_continuous/optlib.py`, `tracks/F_compiler/passes.py` and the xv6 patch, without AI.
5. **Publish**: create a public GitHub repository and push (`git remote add origin <url>`, `git push -u origin main`).
6. Mark your own edits with commits prefixed `[manual]`.

## Rules followed

- Commit prefix `[AI:claude-code]` for AI-generated commits; `[manual]` for your own edits.
- Every reported number comes from a script in the repository; first-version mistakes are kept in the failure log rather than hidden.
- Stochastic methods: repeated runs with fixed seeds, mean and spread reported. Comparisons use the same data, budgets and stopping rules, with a random or naive baseline where meaningful.
- No secrets or private data in the repository.
