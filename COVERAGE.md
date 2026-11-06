# Chapter 6 Coverage Checklist

Maps every Chapter 6 topic to an experiment, code location and result.
**Status:** `TODO` / `WIP` (partly covered) / `DONE` (demonstrated with code and a measured or tested result). **Source:** `PPT n` = in your lecture slides; `PDF` = required by the assignment PDF but not in the slides (confirm with your instructor whether it was taught; the PDF says "where covered").

## PPT 1 - ML optimization

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| ML as optimization (parameters, loss, regularization L0/L1/L2) | PPT1 s5, s27 | tracks/A_continuous/problems.py; exp4 (L2 sweep) | exp4 | DONE |
| Loss functions (MSE, cross-entropy, hinge) | PPT1 s4 | Track A problems.py; Track B dl_discrete.py (hinge) | hinge 0.855 vs logistic 0.856 | DONE |
| Convex vs non-convex landscape, local/global minima, saddle point (each on a small example) | PPT1 s6-7 | experiments.py exp1 | fig1 | DONE |
| Gradient descent, learning-rate sensitivity | PPT1 s8-9 | optlib.gd; exp3; tests slide8/9 | fig3 | DONE |
| SGD and mini-batch SGD (noisy objective) | PPT1 s10 | exp3, exp5 | fig5 | DONE |
| Backpropagation (distinct from optimizer) | PPT1 s11-12 | problems.MLP.grad; exp4 | grad-check 8e-10 | DONE |
| Momentum, Nesterov (NAG) | PPT1 s14-15 | optlib.momentum/nag; exp3 | tests slide14/15 | DONE |
| AdaGrad, RMSProp, Adam | PPT1 s16-18 | optlib; exp3 | tests slide16-18 | DONE |
| Newton's method | PPT1 s19 | optlib.newton; exp1, exp3 | tests slide19 | DONE |
| Gauss-Newton | PDF | optlib.gauss_newton; exp7 | 6 iterations | DONE |
| Curvature approximations (Fisher, diagonal/low-rank, Hessian-vector products) | PDF (PPT1 s35 research only) | exp6 | HVP, Fisher, rank-r, Newton-CG | DONE |
| Convergence: convex O(1/k), strongly convex linear, non-convex stationary | PPT1 s20 | exp2 | fig2 | DONE |
| Optimizer / step-size tuning | PPT1 s21 | exp3 (25-point grid) | fig3 | DONE |
| Continuous vs discrete optimization | PPT1 s23 | A (continuous), B (discrete) | both families implemented and compared | DONE |
| Greedy forward selection / backward elimination (feature selection) | PPT1 s24-25 | featsel.py; exp_featsel | 114 / 455 fits, test acc 0.961 / 0.968 | DONE |
| ILP (knapsack), Branch-and-Bound | PPT1 s26 | knapsack.py; exp_knapsack | B&B nodes 55-150k; ILP = DP optimum | DONE |
| Dynamic programming | PDF | knapsack.dp, scheduling.weighted_dp | exact; = ILP | DONE |
| Sparse learning L0/L1 (Lasso soft-threshold) | PPT1 s27 | Track B: exhaustive k-subset (L0), Lasso path (L1), slide-27 test | L0 27,405 fits vs L1 28 fits, equal accuracy | DONE |
| Discrete choices in deep learning (NAS, quantization, pruning) | PPT1 s28 | Track B dl_discrete.py | pruning+finetune 0.957 at 50%; int8 no loss; NAS greedy = exhaustive | DONE |
| QUBO / Ising (binary-to-spin conversion, penalty terms) | PPT3 s23, s28 + PDF | qubo.py; exp_qubo_penalty | energies equal over all 2^n; feasible only for P >= 0.2 | DONE |
| Submodularity, diminishing returns, testing the property | PPT1 s29-30 | dlib.is_submodular_on_samples | 0 violations; |S|^2 flagged | DONE |
| Greedy (1-1/e) guarantee, checking its assumptions | PPT1 s31 | exp_submodular; Track E | worst ratio 0.953 >= 0.632 | DONE |
| Submodular applications (summarization, sensor placement) | PPT1 s32-33 | dlib coverage (sensor placement); Track E surrogate | greedy/opt 0.994-0.998 | DONE |

## PPT 2 - Parallel optimization

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| Speedup, efficiency | PPT2 s6 | exp_parallel | measured up to 4.25x | DONE |
| Amdahl's and Gustafson's laws | PPT2 s7-8 | dlib.amdahl/gustafson | slide values tested; fitted s=0.17 | DONE |
| Makespan; static vs dynamic scheduling | PPT2 s9-12 | exp_parallel | static 2.37x vs LPT 4.25x at p=12 | DONE |
| Work stealing | PPT2 s13-14 | makespan_work_stealing | 15 -> 11 | DONE |
| Data locality, cache blocking, loop order | PPT2 s16-20 | locality_sync.c | 56x loop order, 7.1x blocking | DONE |
| False sharing, NUMA | PPT2 s20-21 | locality_sync.c | 2.2x from padding; NUMA not measurable on one socket | DONE |
| Synchronization (barrier, lock, atomic, reduction), races, deadlock | PPT2 s23-24 | locality_sync.c | reduction 125x faster than critical; race lost updates 5/5; deadlock not demonstrated | DONE |
| MPI collectives, communication cost T = alpha + n/beta | PPT2 s25-26 | parallel_extras.exp_comm (pipes, not MPI) | alpha 48 us, batching 45x | DONE |
| Overlap, local SGD, sync vs async SGD | PPT2 s27-28 | FedAvg local steps (experiments.py) | E=20: 9 rounds vs 36; async not run | DONE |
| Data / model / pipeline parallelism | PPT4 s10, s14 | exp_parallel | data-parallel 1.6x at p=8; pipeline simulated | DONE |

## PPT 3 - Quantum

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| Qubit, Born rule, shots (finite-shot cost) | PPT3 s5-7 | qsim.sample_mean; exp_shots | std = sqrt(var/S); >=1000 shots ~ exact | DONE |
| Gates, Hadamard interference, CNOT, Bell state | PPT3 s10-17 | tracks/I_quantum_algorithms_qml/tests (Bell, HH, CNOT in Track C) | tests pass | DONE |
| Deutsch-Jozsa / Deutsch | PPT3 s13 | qalgs.deutsch_jozsa; Track I | 20/20, 1 query vs 513 | DONE |
| Bernstein-Vazirani | PDF | qalgs.bernstein_vazirani | exact, 1 query vs n | DONE |
| Grover search (vs classical, query counts) | PPT3 s14 | qalgs.grover; exp_algorithms | queries 100 vs 8192 expected (N=16384); sim slower in wall time | DONE |
| Quantum Fourier Transform, Phase Estimation, Shor | PDF | qalgs.qft/qpe/shor_order_finding | QFT = DFT; Shor N=15 45-58% per attempt (permutation oracle) | DONE |
| Quantum annealing and QUBO embedding limits | PDF | Track C exp_embedding + qalgs.anneal | chain strength trade-off; P(opt)=0.91 at T=30 | DONE |
| NISQ, variational loop (VQA) | PPT3 s19-20 | qopt.py; exp_vqe | SIMULATED | DONE |
| VQE | PPT3 s22, 24, 29 | exp_vqe; test slide22/24/28 | error 1e-11 at 16 params; slide QUBO -4.000 | DONE |
| QAOA on Max-Cut (solves the Track B QUBO) | PPT3 s27-30 | exp_qaoa; test slide30 | ratio 0.83/0.88/0.91 (p=1/2/3) | DONE |
| Parameterized circuits / QNN / variational classifier | PPT3 s18, 25 | hea_ops; test slide25 | slide QNN only; classifier in Track I | DONE |
| Parameter-shift rule | PPT3 s24 | qsim.param_shift_grad | = finite diff 1e-7 | DONE |
| SPSA, quantum natural gradient | PDF | qopt.spsa / qng; exp_vqe | SPSA 5/5; QNG fewest iterations, most circuits | DONE |
| Quantum kernels | PDF | qml.kernel_matrix | moons 0.835, circles 0.949 | DONE |
| Barren plateaus, noise | PPT3 s32 | exp_barren, exp_noise | slope -0.68; noise erases p=3 gain at 3% | DONE |
| QML vs classical comparison | PPT3 s33 | exp_qml | classical RBF SVM / MLP win | DONE |

## PPT 4 - Strategy selection

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| Decision guide (problem type, size, convexity, hardware, budget) | PPT4 s3-7 | Track E strategy table | used to pick greedy/ILP/SA | DONE |
| Model-fit check; data / model / pipeline parallel choice | PPT4 s8-10 | parallel_extras.training_memory_gb | 16/160 GB | DONE |
| Ring all-reduce cost | PPT4 s11 | parallel_extras.ring_allreduce_bytes_per_worker | 700 MB, 70 ms (test) | DONE |
| Amdahl applied to training | PPT4 s13 | tests/test_extras.py; exp_parallel fit | 4.71x; fitted s=0.17 | DONE |
| Pipeline bubble formula | PPT4 s14 | dlib.pipeline_makespan | simulation = formula | DONE |
| Top-k sparsification, communication reduction | PPT4 s15 | parallel_extras.exp_topk | error feedback restores loss at 2% volume | DONE |
| Hybrid-parallel layout search | PPT4 s16 | parallel_extras.hybrid_layouts | 28 layouts (count only, no search) | DONE |
| Parallel hyperparameter search (grid / random / successive halving) | PPT4 s17-18 | parallel_extras.exp_hpo | SH 108 epoch units vs 2187 | DONE |
| Greedy feature selection cost (955 fits) | PPT4 s21 | Track B test + featsel | 955 / 114 fits | DONE |

## Required by the PDF but absent from the four PPTs

| Topic | Track | Note |
|---|---|---|
| FedAvg vs centralized | D | exp_fedavg | E=20: 9 rounds vs 36 for E=5 | DONE |
| MGDA, GradNorm (two-task model) | D | exp_multitask | mgda_norm best mean 0.30 | DONE |
| Gauss-Newton, Fisher/Hessian-vector products | A | Same. |
| SPSA, quantum natural gradient | C | Same. |
| QFT, QPE, Shor, Bernstein-Vazirani, quantum annealing | I | Slides say PPT 3 is "optional enrichment (not in the syllabus)". |
| Dynamic programming | B | Not a dedicated PPT section. |

> Default plan: cover everything above. It is cheap to include and the PDF names these topics. If your instructor says some were not taught, mark them `N/A (not taught)`.

## Track F (compiler) - decisions solved with Chapter 6 methods

| Decision | Method | Where | Result |
|---|---|---|---|
| Pass ordering | greedy vs exhaustive (24 orders) | tracks/F_compiler | greedy optimal in 3/6, 10 vs 96 pass runs |
| Register allocation | greedy colouring vs ILP | tracks/F_compiler | greedy = exact on all graphs |

## Track H (DBMS) - decisions solved with Chapter 6 methods

| Decision | Methods | Where | Result |
|---|---|---|---|
| Index selection under a storage budget | greedy, DP knapsack, interaction-aware ILP | tracks/H_dbms | ILP best or tied at 3 budgets; additive methods caused an 11x regression at 30% |
| Join order | DP over subsets, greedy vs all 24 measured orders vs SQLite | tracks/H_dbms | DP = measured best (0.041 s); worst 15.6 s |

## Track J (software engineering) - decisions solved with Chapter 6 methods

| Decision | Methods | Where | Result |
|---|---|---|---|
| Test-suite minimization (weighted set cover) | greedy vs ILP vs random | tracks/J_software_engineering | 654 -> 61 tests, 100% coverage, 30/33 mutants kept, 1.74x end to end |
| Test prioritization | greedy additional coverage | tracks/J_software_engineering | APFD 0.962 vs 0.743 random |

## Track G (xv6) - OS concept: CPU scheduling

| Feature | Objective | Where | Result |
|---|---|---|---|
| Lottery scheduler + settickets/getpinfo | proportional share | tracks/G_xv6 (xv6_scheduling.patch) | CPU shares 0.49/0.34/0.17 for 30/20/10 tickets |
| MLFQ (3 levels, boost, preemption) | response / turnaround | tracks/G_xv6 | interactive delay 0.00-0.02 ticks; turnaround 1-7% below RR |
| Correctness | usertests -q | results/usertests_policy*.txt | ALL TESTS PASSED under all 3 policies |
