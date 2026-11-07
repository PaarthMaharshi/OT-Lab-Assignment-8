# Demonstration script (about 6-8 minutes)

The PDF asks for a short demonstration of the problem, the method, the evidence of improvement and the key lessons. Every command below runs from the repository root and finishes in seconds unless marked; all numbers on screen come from the committed results.

## 0. Before you start (once)
```bash
pip install -r requirements.txt
cd tracks/E_tower_defense && python -m pytest -q tests && cd ../..
```
Open these images in a viewer so you can switch to them quickly:
`tracks/E_tower_defense/results/baseline_map.png`, `tracks/E_tower_defense/results/comparison.png`, `tracks/A_continuous/results/fig1_landscapes.png`, `tracks/C_quantum_optimization/results/fig5_barren.png`.

## 1. The problem (1 min) - Track E
- Say: "Tower defense: a budget buys towers; where they go decides how much of a 40-enemy wave is destroyed."
- Run: `cd tracks/E_tower_defense && python run_baseline.py` (prints the random-placement statistics).
- Point at: random layouts destroy 1757 +- 954 of 2860 HP at budget 30 and about 18 of 40 enemies leak. The huge spread is the evidence that placement matters.
- Show `results/baseline_map.png` (the path and the best random layout).

## 2. Formulation and method (1.5 min)
- Binary variable per (cell, tower type); one tower per cell; a knapsack budget; objective = damage in the simulator.
- Three methods: greedy and ILP on a fast surrogate (a max-flow model, submodular), simulated annealing on the real simulator. Show the strategy-selection table in `tracks/E_tower_defense/README.md`.
- One sentence on the guarantee: "the budget is a knapsack constraint, so the bound is (1/2)(1-1/e), not 1-1/e, and only on the surrogate."

## 3. Evidence of improvement (1.5 min)
- Show `results/comparison.png` and the table in `results/comparison.md` (`python compare.py` regenerates it in about 2 minutes on 12 cores; do not run it live).
- Say: "Every optimizer beats random by a wide margin; annealing is best in 11 of 12 cases; ILP is exact for its model but trails by up to 17%, because the surrogate ignores slow towers."
- The story of the mistake: the first surrogate let one gun "kill" all enemies, and greedy scored below random. Show the entry in `docs/failure_log.md` (row 1).

## 4. One surprising result from another track (1.5 min) - pick one
- **Track H (database):** adding indexes chosen by a simple knapsack model made the SQLite workload 11x slower (a query-planner regression); the interaction-aware ILP set saved time. Table in `tracks/H_dbms/README.md`.
- **Track J (tests):** the minimised networkx suite (61 of 654 tests) was 2x *slower* when run naively, 1.74x faster with a collect-once plugin, and lost 3 of 33 seeded faults. Table in `tracks/J_software_engineering/README.md`.
- **Track G (xv6):** lottery scheduling delivers 0.49/0.34/0.17 CPU shares for 30/20/10 tickets; the first MLFQ was too slow to react to interactive processes, found by measuring and fixed. Table in `tracks/G_xv6/README.md`.

## 5. Quantum and parallel, honestly (1 min)
- Show `tracks/C_quantum_optimization/results/fig5_barren.png`: gradient variance halves per added qubit (barren plateau).
- Say: "QAOA reaches 0.83 -> 0.91 of the max cut, but classical annealing solves these graphs in milliseconds, and the QML classifiers lose to an SVM. Measured parallel speedup on this laptop is 3-4x, limited by physical cores."

## 6. Lessons (1 min)
1. A surrogate or proxy can be the real bottleneck (E, H, J: models over-predict).
2. Measure the noise before claiming a speedup (F: A/A control; G: QEMU warm-up).
3. A fair comparison needs the same budget, tuning effort and a random baseline (C, I).
4. AI-generated code was wrong 18 times in this project; tests against slide examples and brute force caught most of it.

## Recording tips
Windows: Win+G (Xbox Game Bar) records a window; or OBS Studio. Keep it under 8 minutes; zoom the terminal font to 16-18 pt.
