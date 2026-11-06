# Track E - Main Case Study: Tower-Defense Placement

**Process.** A 2D grid with an enemy path. A budget buys towers (gun: cost 10, range 2.5, 5 damage/tick; cannon: cost 25, range 2, 16 damage/tick; frost: cost 15, slows enemies in range by 50%). Where the towers go decides how much of a 40-enemy wave (fast/weak, normal, slow/tanky) is destroyed before it leaks out. Evidence it needs optimizing: random placements at budget 30 destroy only 1757 +- 954 of 2860 HP and let about 18 of 40 enemies through, with a huge spread; good placements destroy almost everything.

Run: `python compare.py` (both maps, about 2 minutes on 12 cores), `python run_qubo.py`, `python parallel_eval.py`, `python run_baseline.py`. Tests: `python -m pytest tests` (19 tests).

## Formulation
- **Decision variables:** x[c,t] in {0,1}: tower type t on buildable cell c.
- **Constraints:** at most one tower per cell; sum of costs <= budget B (a knapsack constraint).
- **Objective (true):** total damage dealt in the deterministic wave simulator `td/sim.py` (overkill not counted).
- **Surrogate (for greedy / ILP):** a max-flow model: each tower can damage each enemy for its dwell time in range, but fires at one target per tick (per-tower capacity), and an enemy cannot lose more than its HP. Monotone submodular; ignores slows and exact targeting order.
- **QUBO (bridge to Tracks B/C):** one tower type, exactly k towers: E(x) = -sum v_c x_c + sum overlap_cc' x_c x_c' + P (sum x - k)^2 (second-order inclusion-exclusion of path coverage).
- **Success metric:** true simulated damage (and leaked enemies) on the same maps and waves for every method; runtime.

## Strategy selection

| Aspect | Value here | Consequence |
|---|---|---|
| Variable type | binary (cell x type) | discrete methods (Chapter 6 discrete / submodular) |
| Problem size | 70-74 buildable cells x 3 types | exact ILP feasible on a surrogate; exhaustive search on the true objective is not |
| Objective | black-box simulator (non-smooth, no gradient) | gradient methods do not apply; surrogate + exact, or metaheuristic on the simulator |
| Exact or heuristic | both, compared | greedy and ILP on the surrogate; simulated annealing on the simulator |
| Hardware | 12 logical processors | annealing runs are independent: parallel (measured below) |
| Quantum | QUBO with 70 variables | QAOA only on a 14-cell sub-problem (simulation needs 2^n amplitudes) |

## Results

**Main comparison** (true damage; 3 waves x 2 budgets x 2 maps; annealing: 5 seeds, 800 simulator evaluations each; random: 100 placements). Full table: `results/comparison.md`.

| Map | random (mean) | greedy | ILP | annealing (mean) | annealing best or tied |
|---|---|---|---|---|---|
| snake 15x11 (6 cases) | 882-1864 | 2165-3200 | 2172-3325 | 2193-3297 | 5 of 6 (ILP wins wave 2 / budget 30) |
| spiral 13x11 (6 cases) | 878-1884 | 2180-3300 | 2180-2700 (3100 at B=30) | 2185-3309 | 6 of 6 |

- Every optimizer beats random placement by a wide margin (e.g. snake, wave 0, budget 20: random 1048 +- 1069, greedy 2335, ILP 2555, annealing 2564).
- **The surrogate is the bottleneck, not the algorithm.** ILP is optimal for the surrogate yet loses to plain greedy on two spiral cases (2392 vs 2565; 3100 vs 3300) and trails annealing by up to 17% (snake wave 1, budget 30: 2172 vs 2605), because the surrogate ignores slows and targeting. Annealing optimises the true objective directly, at about 10 s per run versus 0.3-2 s for greedy / ILP.
- The first surrogate was badly wrong (one gun "killed" every enemy; greedy then scored *below random*); see `prompts/E/02_optimizers.md`.

**Guarantee check.** The surrogate is monotone submodular (tested on random sets), but the budget is a knapsack constraint, so the applicable bound for cost-benefit greedy plus best single item is (1/2)(1-1/e) = 0.316, not 1-1/e, and only for the surrogate. Measured: greedy reached at least 0.316 of the surrogate optimum on every test instance.

**QUBO view** (`results/qubo.json`). Small instance (14 candidate cells, 3 guns, penalty P = 1.2 x the largest tower value): exact minimum -165 by enumerating 2^14 states; simulated annealing finds it in 10 runs. QAOA (exact simulation): approximation ratio 0.913 / 0.943 / 0.961 and P(optimal) 1.1% / 1.6% / 2.1% for p = 1 / 2 / 3 (uniform sampling: 0.006%); P(exactly 3 towers) 36% / 50% / 64% vs 2.2% for uniform. On the full map (gun only, true simulator, mean of 3 waves) the QUBO+SA placement is *worse than random* (2460 vs 2643 at budget 30; 2875 vs 2928 at budget 40), annealing on the simulator is best (2962 / 2987) and ILP second (2755 / 2945): the pairwise-overlap QUBO is a cruder model than the flow surrogate. The quadratic form fits quantum hardware, not the problem.

**Parallel evaluation** (`results/parallel_eval.json`, 12 independent annealing runs): speedup 1.75 / 2.75 / 3.01 / 2.62 with 2 / 4 / 6 / 12 processes (efficiency 0.88 -> 0.22), identical results in every case. The peak at 6 matches Track D (likely 6 physical cores behind 12 logical processors).

## Trade-offs and when methods fail
- Greedy: fastest (0.3-0.7 s), no tuning, but blind to slows and tower interactions; stops using budget once the surrogate saturates.
- ILP: exact for its model, sensitive to model error; cost grows with candidate count (SCIP solved all instances in under 5 s).
- Simulated annealing on the simulator: best quality, about 15x more time, stochastic (std up to 106 damage), and its cost is dominated by simulator calls, which parallelise to about 3x here.
- Random placement is a necessary baseline: its spread (sd about 1000 of 2860) shows why optimising matters at small budgets, while budgets of 40+ saturate (every sensible layout kills everything).

## How the approach would adapt
Bigger maps or more tower types: keep greedy/ILP on a better surrogate (e.g. include slows) as a warm start for annealing; parallelise annealing across cores or machines. Stochastic waves: optimise the mean over sampled waves (sample-average approximation). Real-time play: greedy only.

## Limitations
Synthetic maps and waves; a deterministic simulator; 3 waves x 2 budgets x 2 maps; annealing hyperparameters (temperatures, 800 evaluations) were not tuned per map.
