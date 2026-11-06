# Track I - Quantum Algorithms and QML versus Classical

**Everything here is simulated** on the numpy simulator from Track C (`../C_quantum_optimization/qsim.py`); no real quantum hardware was used. Every algorithm's circuit was written gate by gate for this track (`qalgs.py`); no library implementation is called (PennyLane was used only for a cross-check in Track C).
Run: `python experiments.py` (about 2 minutes). Tests: `python -m pytest tests` (9 tests, all checked against closed forms or independent computation).

| Taught algorithm | Code | Verified by |
|---|---|---|
| Deutsch-Jozsa | `deutsch_jozsa` (one oracle call) | 20/20 correct for n = 2..10; constant -> P(0..0) = 1, balanced -> 0 |
| Bernstein-Vazirani | `bernstein_vazirani` (CNOT oracle, gate level) | secret recovered with probability 1 up to n = 12 |
| Grover | `grover` | success probability equals sin^2((2k+1) asin(1/sqrt N)) to 1e-10; N=4 -> 1 round gives certainty (slide 14) |
| QFT | `qft`, `qft_matrix` | equals the DFT matrix to 1e-12 for t = 1..5; inverse undoes it |
| Phase estimation | `qpe` | phi=1/8, 3 bits -> exact with probability 1; phi=0.3, 6 bits -> 0.297 with probability 0.875 (bound 4/pi^2 = 0.405) |
| Shor (N = 15) | `shor_order_finding`, `order_from_measurement` | orders of 2, 4, 7, 8, 11, 13 found; see below |
| Quantum annealing (simulated) | `anneal` | adiabatic evolution on a Track B Max-Cut graph |
| VQE / QAOA / parameter-shift / SPSA / QNG | Track C | - |
| Variational classifier, quantum kernel | `qml.py` | gradient equals finite differences; kernel is PSD with unit diagonal |

## Query complexity and the cost of simulating it

| Algorithm | quantum queries | classical queries | note |
|---|---|---|---|
| Deutsch-Jozsa, n = 10 | 1 | 513 (worst case, deterministic) | randomized classical needs only a few queries with tiny error |
| Bernstein-Vazirani, n = 12 | 1 | 12 | |
| Grover, N = 16384 | 100 | 8,192 expected (16,384 worst) | quadratic, not exponential: ratio grows 2.5x (N=4) to 82x (N=16384) |

**But in wall-clock time simulation has the opposite sign:** Grover at n = 14 took 0.24 s on the simulator versus 0.7 ms for a Python linear search. Query counts are theoretical advantages; the exponential cost of simulating 2^n amplitudes is the price of studying them on a laptop. Grover over-rotates: for n = 6 success peaks at the optimal 6 iterations and falls afterwards.

## Shor, N = 15
The controlled modular multiplications are applied as **exact permutations** of the 4-qubit work register (not decomposed into gates), so no gate-count or resource claim is made. With strict classical post-processing (no search over multiples of the period), one attempt recovers the correct order in 90-116 of 200 shots (45-58%) for all six bases and each of those yields a nontrivial factor of 15. About half of the attempts are uninformative by design (measurement 0 or a half-period), so the algorithm is repeated. N = 15 is trivial classically; nothing here shows or implies an advantage.

## Annealing (simulated closed system, n = 8 Max-Cut graph)
Probability of ending in an optimal cut: 0.011 (T = 0.5), 0.067 (2), 0.34 (8), 0.91 (30), 1.00 (100). The smallest gap between the two lowest levels of the symmetric sector along the schedule is 0.43, consistent with needing a total time of tens of units. This is an idealised noiseless evolution; Track C shows the embedding limits of real annealers.

## Hardware noise model (simulated)
Bernstein-Vazirani with the all-ones secret under single-qubit depolarizing noise after every gate (density-matrix simulation): success probability at 1% noise falls from 0.942 (n = 2, 8 gates) to 0.847 (n = 6, 20 gates); at 3% noise from 0.837 to 0.609. The one-query advantage (n queries -> 1) is bought with a circuit whose reliability degrades as n grows.

## QML versus classical (300 points, 50/50 split, 5 seeds, 2 features, quantum models on 2 simulated qubits)

| Model | moons acc | circles acc | parameters | circuit evaluations | train time (moons) |
|---|---|---|---|---|---|
| Variational classifier (3 layers, data re-uploading, parameter-shift + Adam) | 0.920 +- 0.023 | 0.919 +- 0.067 | 12 | 225,000 | 0.74 s |
| Quantum kernel + SVM (ZZ feature map) | 0.835 +- 0.014 | 0.949 +- 0.023 | about 35-62 (support vectors) | 11,325 | 0.03 s |
| Logistic regression | 0.851 +- 0.032 | 0.491 +- 0.016 | 3 | 0 | 0.005 s |
| RBF SVM | **0.945 +- 0.015** | 0.988 +- 0.008 | about 49-79 | 0 | 0.001 s |
| MLP (8 hidden) | 0.837 +- 0.016 | 0.643 +- 0.165 | 33 | 0 | 0.20 s |
| MLP (16,16) | 0.873 +- 0.052 | **0.991 +- 0.003** | 337 | 0 | 0.22 s |

**Honest result: the quantum models are not better.** The best classical model beats both quantum models on both datasets (RBF SVM 0.945 vs 0.920 on moons; MLP 0.991 vs 0.949 on circles), at roughly 1000x less training time. The variational classifier does use far fewer parameters (12) than the MLP (337) and beats logistic regression and the small MLP on moons, and the quantum kernel beats them on circles; but it needs 225,000 circuit executions, which on real hardware (shots, queueing) would cost orders of magnitude more than the milliseconds here. The 2-feature, 2-qubit setting also removes the data-loading and barren-plateau problems that dominate at scale, so this test cannot support an advantage claim either way.

## Contrast with parallel computing (Track D)
Track D measured real speedups of 3-4x from multiprocessing on this machine (and a slowdown for a memory-bound step); the quantum entries above are **theoretical query counts** or **simulations that run slower than the classical code**. A measured parallel speedup is hardware-bound but real; a quantum advantage is an asymptotic statement about hardware that is noisy and small today.

## When quantum helps and when it does not (summary of the evidence here)
Helps in principle: query problems with structure (DJ, BV: exponential/linear separations vs deterministic classical; Grover: quadratic). Does not help here: classification on small low-dimensional data (classical wins), anything needing many circuit evaluations or deep circuits under noise, and wall-clock on a simulator.

## Limitations
Simulation only; N = 15 for Shor with permutation oracle; no error mitigation; 2-qubit QML only; classical baselines lightly tuned (RBF SVM with C = 10, MLP with fixed sizes); parameter counts for SVMs are support-vector counts times 3 and are approximate.
