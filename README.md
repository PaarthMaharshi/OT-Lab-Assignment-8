# Optimize Anything - 14-Day Vibe-Coding Challenge (Assignment 8)

Course: Optimization Techniques (R5CO3101T). All ten tracks (A-J) are compulsory.
Main case study (Track E): **Tower-defense placement optimization**.

## Tracks

| Track | Topic | Folder | Status |
|---|---|---|---|
| A | Continuous optimization (GD ... Adam, Newton, backprop, landscapes) | `tracks/A_continuous` | WIP |
| B | Discrete optimization (greedy, ILP, B&B, DP, QUBO/Ising) | `tracks/B_discrete` | WIP |
| C | Quantum optimization (VQE, QAOA, parameter-shift, SPSA, QNG) | `tracks/C_quantum_optimization` | WIP |
| D | Submodular, parallel, distributed (FedAvg, MGDA, GradNorm) | `tracks/D_submodular_parallel_distributed` | WIP |
| E | **Main case study: tower defense** | `tracks/E_tower_defense` | WIP |
| F | Compiler optimization | `tracks/F_compiler` | WIP |
| G | xv6 OS optimization | `tracks/G_xv6` | TODO |
| H | DBMS optimization | `tracks/H_dbms` | WIP |
| I | Quantum algorithms and QML vs classical | `tracks/I_quantum_algorithms_qml` | TODO |
| J | Software-engineering optimization | `tracks/J_software_engineering` | TODO |

Chapter 6 coverage: see [COVERAGE.md](COVERAGE.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Environment notes: xv6 needs QEMU (RISC-V) and a RISC-V toolchain, best under WSL. Track H needs PostgreSQL or SQLite. Track F can use GCC or a Python mini-compiler. Record hardware (CPU model, cores, RAM, OS) in each track's results.

## AI tools register (required)

| Tool | Model / version | Tracks | Dates | Kind of work |
|---|---|---|---|---|
| Claude Code | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | setup | 2026-11-06 | planning, scaffolding, documentation |
| _add every tool you use, including for debugging or explanation_ | | | | |

## Rules we follow

- Commit prefix `[AI:tool]` for AI-generated commits; `[manual]` for your own edits.
- One `prompts/<track>/` folder per track: original prompt, AI response excerpt, revised prompt, commit link, verification note, chat export. See `prompts/TEMPLATE.md`.
- `docs/failure_log.md`: at least 3 AI errors with fix and lesson.
- `docs/tool_comparison.md`: at least 2 AI tools on one task.
- Remove API keys, tokens and private data before publishing.
- Stochastic methods: repeated runs with fixed seeds, report mean and spread.
- Viva: you must be able to explain and modify every file without AI.
