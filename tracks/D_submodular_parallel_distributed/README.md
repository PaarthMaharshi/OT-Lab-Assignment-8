# Track D - Submodular, Parallel and Distributed Optimization

Run: `python experiments.py` (about 3 minutes; the parallel part takes 2+ minutes). Tests: `python -m pytest tests` (8 tests; slide values for makespan 15 / 9 / 11, Amdahl 6.4, Gustafson 14.5, pipeline bubble 3/11).
Parallel numbers are **measured** here with `multiprocessing` on 12 logical processors; pipeline parallelism is **simulated**; FedAvg and multi-task results use small numpy models.

| Requirement | Code | Result |
|---|---|---|
| Greedy on a submodular objective, conditions checked | `dlib.greedy_cover`, `opt_cover` (ILP), `is_submodular_on_samples` | max coverage: 0 diminishing-returns violations; greedy/optimum mean 0.994-0.998, worst 0.953 (bound 0.632). A supermodular f(S)=\|S\|^2 is flagged (300/300 violations), so the bound would not apply |
| Load balancing, makespan | `makespan_*`, `exp_parallel` | tables below |
| Work stealing | `makespan_work_stealing` | slide example 15 -> 11; random tasks: stealing 29.0 vs static 35.6 (p=4), 17.6 vs 22.7 (p=8), lower bound 26.2 / 13.0 |
| Speedup, efficiency, Amdahl / Gustafson | `exp_parallel` | implied serial fraction 0.17 from the p=12 measurement |
| Communication overhead | tiny tasks, p=8 | chunksize 1: speedup 1.46; chunksize 750: 3.99 |
| Data parallelism | sharded logistic gradient, 2M samples | speedup 1.12 (p=2), 1.39 (p=4), 1.61 (p=8), **0.83 (p=12)** |
| Pipeline parallelism | `pipeline_makespan` simulation | bubble equals (S-1)/(m+S-1) exactly (S=4: 75% at m=1, 27% at m=8, 8.6% at m=32) |
| FedAvg vs centralized | `fedavg`, `exp_fedavg` | below |
| MGDA, GradNorm vs equal weights | `train_multitask`, `exp_multitask` | below |

**Load balancing (240 tasks, measured speedup over a 2.4 s serial run).** Irregular (log-normal) tasks: static 2.02 / 3.35 / 3.24 / 2.37 at p = 2 / 4 / 8 / 12; dynamic 1.76 / 3.41 / 3.57 / 3.01; dynamic with largest-first 1.74 / 3.59 / 3.71 / 4.25. Static degrades at p=12 because one block gets the heavy tail; largest-first dynamic scheduling is best at high p. At p=2 the pool dispatch overhead makes dynamic slower than static. For uniform tasks static is as good as dynamic (4.05 vs 4.03 at p=8), as the slides predict.

**Why speedup stays at 3-4x.** The machine reports 12 logical processors, probably 6 physical cores with hyper-threading, plus pool overhead. The data-parallel gradient is memory-bound (a 2M x 20 array), so extra processes stop helping and p=12 is slower than serial. This is the limit of parallelism the assignment asks about, not a bug. The Amdahl serial fraction is therefore not a purely algorithmic number: it lumps hardware limits in.

**FedAvg (10 clients, logistic regression; global loss after 60 rounds).**

| Setting | centralized GD | E=1 | E=5 | E=20 |
|---|---|---|---|---|
| IID | 0.412 | 0.468 | 0.415 | 0.412 |
| non-IID | 0.150 | 0.191 | 0.157 | 0.154 |

More local steps per round cut the communication rounds needed (rounds to within 0.01 of centralized, IID: E=5 needs 36, E=20 needs 9) at the price of more client computation (E=20 is 1200 gradient steps per client). Non-IID data needs more rounds (50 for E=5, 24 for E=20). With E=1 and equal-size IID clients one FedAvg round equals one centralized gradient step (exact test).

**Multi-task (two tasks sharing ONE hidden feature; loss relative to each task's starting loss; 5 problems).**

| Weighting | task 1 | task 2 | mean |
|---|---|---|---|
| equal | 0.00005 | 0.721 | 0.361 |
| MGDA (raw) | 0.979 | 0.002 | 0.491 |
| MGDA (normalised gradients) | 0.498 | 0.105 | 0.302 |
| GradNorm | 0.032 | 0.613 | 0.323 |

Equal weights let the large-scale task win; raw MGDA does the opposite, because the min-norm point favours the smaller gradient, so a task with a big gradient barely trains. Normalising the gradients (as in the MGDA literature) fixes that and gives the best mean, but this is a trade-off between tasks, not a universal winner. With a wider shared layer (4 hidden units) the tasks hardly conflicted and equal weighting was best (0.005 / 0.005), so these methods matter only when tasks truly compete.

## Limitations
Single machine with logical processors; GradNorm used alpha=1.5 and a hand-picked weight learning rate; the two-task problem is synthetic; model and pipeline parallelism are not measured on real devices.
