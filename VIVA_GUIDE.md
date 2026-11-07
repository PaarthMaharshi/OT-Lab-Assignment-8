# Viva study guide

The PDF says you must explain and modify your code live without AI, and components you cannot explain get no credit. Study in this order; each section lists what to know, likely questions with short answers, and a modification exercise you can practise (check it by running the named tests).

## 1. Track E simulator - `tracks/E_tower_defense/td/sim.py`
**Know:** one tick = spawn -> slow auras -> towers fire -> remove dead and move. A damage tower hits the in-range enemy that is furthest along the path; overkill is not counted (`d = min(dps, hp)`); the strongest slow wins (slows do not stack); an enemy whose position reaches the last path cell has leaked.
- *Why is the simulator deterministic?* Towers are processed in sorted cell order and ties go to the first enemy; no randomness, so every method is scored on identical conditions.
- *What does `validate_placement` enforce?* Buildable cell, known tower type, total cost within budget: the constraints of the formulation.
- **Exercise:** make towers target the enemy with the lowest HP instead of the furthest one (change the `best` comparison in the firing loop). Run `python -m pytest tests -q`: `test_targets_furthest_enemy` still passes (same total), the hand-computed tests still pass. Then explain why the total damage barely changes on these tests.

## 2. Track E optimizers - `td/optim.py`, `td/surrogate.py`
**Know:**
- Surrogate: `dmg[k,e]` = dps x ticks enemy e spends in tower k's range; `cap[k]` = dps x ticks in which *any* enemy is in range (one target per tick). Value = max flow: sum z[k,e] with z <= dmg, sum_e z <= cap, sum_k z <= hp. Monotone submodular.
- Greedy: repeatedly add the (cell, type) with the best gain per cost; finally compare with the best single tower (needed for the knapsack bound 1/2 (1-1/e)).
- ILP: binary x[k], continuous z[k,e] <= dmg x[k], tower capacity, enemy hp, one tower per cell, budget.
- Annealing: moves relocate / change type / add / remove; reject over-budget moves without evaluating; accept worse moves with probability exp(delta / T); T decays geometrically.
- *Why is ILP not optimal for the game?* It is optimal for the surrogate, which ignores slows and targeting order.
- *Why not (1-1/e)?* That bound needs a cardinality constraint; ours is a budget (knapsack).
- **Exercise:** add a cooling-schedule parameter to `anneal` (linear instead of geometric) and compare on one wave with `compare.py`-style code; or change `SA_EVALS` and explain the quality/time trade-off.

## 3. Track A optimizers - `tracks/A_continuous/optlib.py`
**Know the update rules** (all tested against the slides): GD `w -= lr g`; momentum `v = beta v + lr g; w -= v`; NAG uses the gradient at `w - beta v`; AdaGrad divides by sqrt(sum g^2); RMSProp by sqrt of a moving average; Adam adds momentum and bias correction; Newton solves `H d = g`; Newton-CG solves it with conjugate gradients using Hessian-vector products only; Gauss-Newton uses `J^T J`.
- *Why can Newton converge to a saddle?* It finds stationary points of the quadratic model; on an indefinite Hessian the step can point uphill (shown in `fig1_landscapes.png`).
- *Difference between backprop and the optimizer?* Backprop computes the gradient (`MLP.grad`, chain rule); the optimizer decides how to step with it.
- **Exercise:** implement Adam without bias correction and show the first step on f = w^2, w0 = 3 is no longer 2.9 (`test_adam_slide16` will fail; explain why).

## 4. Track F passes - `tracks/F_compiler/passes.py`
**Know:** basic blocks start at labels and after jumps; `cprop` keeps a map of known copies/constants per block and folds `bin` with two constants; `cse` keeps available expressions per block and invalidates any that use a redefined variable; `dce` computes liveness backwards over the CFG and deletes assignments to dead variables (never a division that could trap).
- *Why can't DCE delete `x = a / b` when b might be 0?* It would remove a runtime error, changing behaviour.
- *Why did fewer instructions not always run faster?* At -O0 the CPU hides multiply-vs-shift and cheap instructions are not on the critical path.
- **Exercise:** add the algebraic rule `x * 2 -> x + x` to `strength` (choose it instead of the shift) and run `python -m pytest tests -q`: the fuzz test checks behaviour is unchanged.

## 5. Track G xv6 patch - `tracks/G_xv6/xv6_scheduling.patch`
**Know:** `sched_policy` switches between the original round-robin loop and `pick_proc()`; lottery sums tickets of runnable processes and draws a random winner; MLFQ picks the lowest runnable level, round robin inside it, slices 1/2/4 ticks, demotion when a slice is used up, boost every 50 ticks, yield when a higher-priority process is runnable; `usertrap` counts `rticks` per process; `getpinfo` copies a static `pstat` to user space with `copyout`.
- *Why a static `kpstat`?* The struct is about 2 KB and the kernel stack is 4 KB.
- *Why does scheduling not change makespan?* The CPU never idles; scheduling only changes who waits.
- **Exercise:** change the MLFQ boost interval from 50 to 200 ticks, rebuild (`bash build.sh`), rerun `python bench.py smoke`, and explain the effect on the long jobs.

## Questions the PDF says you must answer (short versions; full answers in REPORT.md)
1. Process and evidence: random placements destroy 1757 +- 954 of 2860 HP at budget 30.
2. Formulation: binary x[cell,type], one per cell, budget, maximise simulated damage.
3. Methods and why: greedy (fast, guarantee on surrogate), ILP (exact for a model), annealing (real objective), QUBO/QAOA (quantum bridge).
4. Result: annealing best in 11 of 12 cases; everything beats random.
5. Trade-offs: surrogate error, run time, stochasticity; penalty sizing; quantum loses at this size.
6. Correctness: 106 tests, usertests, identical outputs/results checks, fixed seeds.
7. Scaling: warm-start annealing with greedy, parallelise, sample-average over random waves.
8. Compiler / xv6 / DBMS: what was implemented and how correctness was shown (see each README).
9. Quantum vs parallel: query advantages are real but simulated wall time is worse; parallel speedups are real but bounded at 3-4x here.
10. AI tools: Claude Code (Sonnet 5.5, Opus 5.5), Haiku 5.5 in the comparison; 18 documented errors.

## How to practise
Close every AI tool, open one file, and explain it aloud function by function; then do its exercise and run the tests. If you cannot explain a part, read it until you can - or simplify it and commit the change yourself as `[manual]`.
