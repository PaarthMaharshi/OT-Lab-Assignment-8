# Track E - Tower Defense Placement (main case study)

**Process:** a 2D grid map with an enemy path. A fixed budget buys towers. Where we place them decides how much damage the waves take before leaking through.

## Draft formulation (to be refined)

- **Decision variables:** x[c,t] in {0,1}: tower type t is built on buildable cell c.
- **Constraints:** sum_t x[c,t] <= 1 per cell; sum cost[t]*x[c,t] <= B (budget).
- **Objective:** maximize expected damage dealt to a wave while it walks the path (equivalently minimize enemies leaked).
  - **Simplified (ILP/greedy) objective:** path-cell coverage weighted by tower damage per second times time-in-range. Coverage-style, so submodular when each path cell counts once.
  - **True objective:** full simulator score (enemy HP, speed, slows, targeting).
- **Baselines:** random valid placement; hand-designed/"expert" placement.
- **Metrics:** leaked enemies, total damage, score per budget, runtime.

## Draft strategy-selection table

| Aspect | Value | Consequence |
|---|---|---|
| Variable type | binary | discrete methods |
| Problem size | e.g. 100-2000 cells x 3 types | ILP exact up to medium; heuristics beyond |
| Exact or heuristic | both, compare | ILP (OR-Tools) vs greedy vs simulated annealing (+ QUBO as bridge to Track B/C) |
| Hardware | 12-core CPU | simulator evaluations parallelize across processes |
| Parallelism | independent placements/seeds | embarrassingly parallel evaluation |

## Methods to compare (at least two required)

Random, hand-designed, greedy (marginal gain), ILP, simulated annealing. Check the submodularity of the simplified objective before claiming the (1-1/e) greedy bound; slowing towers and overlapping damage may break it.

## Results so far (true-simulator damage; 15x11 snake map, 40 enemies, 3 waves)

See `results/comparison.md`, `results/comparison.json`, `results/comparison.png`. Reproduce: `python compare.py` (about 1 min on 12 cores).

| wave | budget | total HP | random (mean±sd, 100 seeds) | greedy | ILP | annealing (5 seeds, 800 sim evals) |
|---|---|---|---|---|---|---|
| 0 | 20 | 2860 | 1048±1069 | 2335 | 2555 | 2564±6 |
| 0 | 30 | 2860 | 1757±954 | 2335 | 2640 | 2847±16 |
| 1 | 20 | 2660 | 882±900 | 2165 | 2180 | 2193±16 |
| 1 | 30 | 2660 | 1534±849 | 2165 | 2172 | 2605±47 |
| 2 | 20 | 3440 | 1073±1098 | 2810 | 2810 | 2811±29 |
| 2 | 30 | 3440 | 1864±1063 | 3200 | 3325 | 3297±106 |

Findings:
- All three optimizers beat random by a wide margin; SA is best or tied in 5 of 6 cases.
- Greedy and ILP optimize a surrogate. At budget 30 the surrogate saturates (one cannon "kills everything"), so they cannot use the extra budget and trail SA by up to 15% (wave 1: 2172 vs 2605). That gap is the cost of the surrogate, not of the algorithm.
- ILP is exact only for the surrogate (`proven_optimal` is in the JSON). It is not optimal for the real game.
- Greedy is ~5x faster than ILP, SA is ~10 s but scores the real objective. All timings are on one 12-core machine; SA runs in parallel processes.
- Guarantee: the surrogate is monotone submodular (max-flow), but the constraint is a knapsack, so the bound is (1/2)(1-1/e), not 1-1/e, and it applies to the surrogate only.
- The first surrogate was wrong (see `prompts/E/02_optimizers.md`); the comparison table exposed it.

## Plan

1. `sim.py` - deterministic wave simulator + tests (known small maps with hand-computed scores).
2. Baselines. 3. Greedy. 4. ILP. 5. Simulated annealing. 6. Fair comparison over maps, sizes and seeds. 7. Plots + short demo (visualised placement).
