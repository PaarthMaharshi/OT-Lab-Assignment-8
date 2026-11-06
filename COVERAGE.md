# Chapter 6 Coverage Checklist

Maps every Chapter 6 topic to an experiment, code location and result.
**Status:** `TODO` / `WIP` / `DONE`. **Source:** `PPT n` = in your lecture slides; `PDF` = required by the assignment PDF but not in the slides (confirm with your instructor whether it was taught; the PDF says "where covered").

## PPT 1 - ML optimization

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| ML as optimization (parameters, loss, regularization L0/L1/L2) | PPT1 s5, s27 | tracks/A_continuous/problems.py; exp4 (L2 sweep) | exp4 | WIP |
| Loss functions (MSE, cross-entropy, hinge) | PPT1 s4 | problems.py: cross-entropy (logistic, MLP); MSE in exp7 | partial: hinge not run | WIP |
| Convex vs non-convex landscape, local/global minima, saddle point (each on a small example) | PPT1 s6-7 | experiments.py exp1 | fig1 | WIP |
| Gradient descent, learning-rate sensitivity | PPT1 s8-9 | optlib.gd; exp3; tests slide8/9 | fig3 | WIP |
| SGD and mini-batch SGD (noisy objective) | PPT1 s10 | exp3, exp5 | fig5 | WIP |
| Backpropagation (distinct from optimizer) | PPT1 s11-12 | problems.MLP.grad; exp4 | grad-check 8e-10 | WIP |
| Momentum, Nesterov (NAG) | PPT1 s14-15 | optlib.momentum/nag; exp3 | tests slide14/15 | WIP |
| AdaGrad, RMSProp, Adam | PPT1 s16-18 | optlib; exp3 | tests slide16-18 | WIP |
| Newton's method | PPT1 s19 | optlib.newton; exp1, exp3 | tests slide19 | WIP |
| Gauss-Newton | PDF | optlib.gauss_newton; exp7 | 6 iterations | WIP |
| Curvature approximations (Fisher, diagonal/low-rank, Hessian-vector products) | PDF (PPT1 s35 research only) | exp6 | HVP, Fisher, rank-r, Newton-CG | WIP |
| Convergence: convex O(1/k), strongly convex linear, non-convex stationary | PPT1 s20 | exp2 | fig2 | WIP |
| Optimizer / step-size tuning | PPT1 s21 | exp3 (25-point grid) | fig3 | WIP |
| Continuous vs discrete optimization | PPT1 s23 | A, B | | | TODO |
| Greedy forward selection / backward elimination (feature selection) | PPT1 s24-25 | featsel.py; exp_featsel | 114 / 455 fits, test acc 0.961 / 0.968 | WIP |
| ILP (knapsack), Branch-and-Bound | PPT1 s26 | knapsack.py; exp_knapsack | B&B nodes 55-150k; ILP = DP optimum | WIP |
| Dynamic programming | PDF | knapsack.dp, scheduling.weighted_dp | exact; = ILP | WIP |
| Sparse learning L0/L1 (Lasso soft-threshold) | PPT1 s27 | exp_featsel (L1 path); test slide27 | L1 test acc 0.967; L0 only as exhaustive search | WIP |
| Discrete choices in deep learning (NAS, quantization, pruning) | PPT1 s28 | not done | listed only | TODO |
| QUBO / Ising (binary-to-spin conversion, penalty terms) | PPT3 s23, s28 + PDF | qubo.py; exp_qubo_penalty | energies equal over all 2^n; feasible only for P >= 0.2 | WIP |
| Submodularity, diminishing returns, testing the property | PPT1 s29-30 | D | | | TODO |
| Greedy (1-1/e) guarantee, checking its assumptions | PPT1 s31 | D, E | | | TODO |
| Submodular applications (summarization, sensor placement) | PPT1 s32-33 | D | | | TODO |

## PPT 2 - Parallel optimization

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| Speedup, efficiency | PPT2 s6 | D | | | TODO |
| Amdahl's and Gustafson's laws | PPT2 s7-8 | D | | | TODO |
| Makespan; static vs dynamic scheduling | PPT2 s9-12 | D | | | TODO |
| Work stealing | PPT2 s13-14 | D | | | TODO |
| Data locality, cache blocking, loop order | PPT2 s16-20 | D | | | TODO |
| False sharing, NUMA | PPT2 s20-21 | D | | | TODO |
| Synchronization (barrier, lock, atomic, reduction), races, deadlock | PPT2 s23-24 | D | | | TODO |
| MPI collectives, communication cost T = alpha + n/beta | PPT2 s25-26 | D | | | TODO |
| Overlap, local SGD, sync vs async SGD | PPT2 s27-28 | D | | | TODO |
| Data / model / pipeline parallelism | PPT4 s10, s14 | D | | | TODO |

## PPT 3 - Quantum

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| Qubit, Born rule, shots (finite-shot cost) | PPT3 s5-7 | C, I | | | TODO |
| Gates, Hadamard interference, CNOT, Bell state | PPT3 s10-17 | I (write circuits by hand) | | | TODO |
| Deutsch-Jozsa / Deutsch | PPT3 s13 | I | | | TODO |
| Bernstein-Vazirani | PDF | I | | | TODO |
| Grover search (vs classical, query counts) | PPT3 s14 | I | | | TODO |
| Quantum Fourier Transform, Phase Estimation, Shor | PDF | I | | | TODO |
| Quantum annealing and QUBO embedding limits | PDF | C, I | | | TODO |
| NISQ, variational loop (VQA) | PPT3 s19-20 | C | | | TODO |
| VQE | PPT3 s22, 24, 29 | C | | | TODO |
| QAOA on Max-Cut (solves the Track B QUBO) | PPT3 s27-30 | C | | | TODO |
| Parameterized circuits / QNN / variational classifier | PPT3 s18, 25 | C, I | | | TODO |
| Parameter-shift rule | PPT3 s24 | C | | | TODO |
| SPSA, quantum natural gradient | PDF | C | | | TODO |
| Quantum kernels | PDF | I | | | TODO |
| Barren plateaus, noise | PPT3 s32 | C | | | TODO |
| QML vs classical comparison | PPT3 s33 | I | | | TODO |

## PPT 4 - Strategy selection

| Topic | Source | Track | Code | Result | Status |
|---|---|---|---|---|---|
| Decision guide (problem type, size, convexity, hardware, budget) | PPT4 s3-7 | E (strategy table) | | | TODO |
| Model-fit check; data / model / pipeline parallel choice | PPT4 s8-10 | D | | | TODO |
| Ring all-reduce cost | PPT4 s11 | D | | | TODO |
| Amdahl applied to training | PPT4 s13 | D | | | TODO |
| Pipeline bubble formula | PPT4 s14 | D | | | TODO |
| Top-k sparsification, communication reduction | PPT4 s15 | D | | | TODO |
| Hybrid-parallel layout search | PPT4 s16 | D (optional) | | | TODO |
| Parallel hyperparameter search (grid / random / successive halving) | PPT4 s17-18 | J | | | TODO |
| Greedy feature selection cost (955 fits) | PPT4 s21 | B / J | | | TODO |

## Required by the PDF but absent from the four PPTs

| Topic | Track | Note |
|---|---|---|
| FedAvg vs centralized training | D | Not in slides. Ask whether it was taught. |
| MGDA, GradNorm (two-task model) | D | Same. |
| Gauss-Newton, Fisher/Hessian-vector products | A | Same. |
| SPSA, quantum natural gradient | C | Same. |
| QFT, QPE, Shor, Bernstein-Vazirani, quantum annealing | I | Slides say PPT 3 is "optional enrichment (not in the syllabus)". |
| Dynamic programming | B | Not a dedicated PPT section. |

> Default plan: cover everything above. It is cheap to include and the PDF names these topics. If your instructor says some were not taught, mark them `N/A (not taught)`.
