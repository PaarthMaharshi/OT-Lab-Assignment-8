# Track C - Quantum Optimization

**Everything here is simulated on a numpy simulator written for this track (`qsim.py`). No real quantum hardware was used, and nothing below is a hardware or speedup claim.**
The simulator is cross-checked against PennyLane (test), and every slide value that can be checked is a test (14 tests).

Run: `python experiments.py [exp_name ...]` (about 8 minutes; VQE tuning is most of it). Tests: `python -m pytest tests`.

| Requirement | Code | Evidence |
|---|---|---|
| Simulator vs hardware | `qsim.py` (statevector + density matrix) | PennyLane cross-check; depolarizing channel checked analytically |
| VQE | `qopt.hea_ops`, `tfim_dense`, exp_vqe | TFIM n=4 (exact E0 = -4.7588): error 2.5e-2 (8 params), 1.0e-4 (12), 1e-11 (16). Slide-28 QUBO with a 12-parameter ansatz reaches -4.000 in 10/10 runs |
| QAOA solving the Track B QUBO | `qaoa_ops`, `optimize_qaoa`, exp_qaoa | slide values 3.237 (p=1) and 3.856 (p=2) reproduced; 12 Track B Max-Cut graphs below |
| PQC / QNN | `hea_ops`; slide-25 one-qubit classifier test | trained variational classifier vs classical model is Track I |
| Parameter-shift | `qsim.param_shift_grad` | equals finite differences (1e-7) incl. shared QAOA parameters; costs 2 executions per gate parameter |
| SPSA | `qopt.spsa` | 2 executions per step, any dimension |
| Quantum natural gradient | `qsim.metric_tensor`, `qopt.qng` | Fubini-Study metric of RY = 1/4 (test) |
| Finite shots | `sample_mean`, exp_shots | estimator std matches sqrt(var/S) |
| Hardware noise (simulated) | `run_dm`, exp_noise | depolarizing density-matrix simulation |
| Barren plateaus | exp_barren | slope of ln Var = -0.68 for a global cost |
| QUBO embedding limits on annealers | exp_embedding | clique-embedding resource table + chain-strength experiment |

## Findings (`results/summary.json`, figures `results/fig1..fig5*.png`)

**QAOA on the 12 Track B graphs (6-12 nodes, exact expectation).** Mean E[cut]/max-cut = 0.833 (p=1), 0.881 (p=2), 0.905 (p=3); random guessing gives 0.650. P(sampling an optimal cut) rises 0.17 -> 0.27 -> 0.35. At fixed p the ratio falls as graphs grow (p=3: 0.936 at n=6, 0.872 at n=12). Honest comparison: "best cut in 1000 shots" is 100% of optimal, but best-of-1000 *uniformly random* bitstrings is also 100% at these sizes (at most 4096 states), so no advantage is shown. Classical simulated annealing (Track B) reaches the optimum in milliseconds; simulating QAOA took 2.1 s (n=6) to 13.4 s (n=12) for the same 3 graphs x p=1..3, growing exponentially.

**Optimizers on VQE (16 parameters; hyperparameters tuned on separate seeds, compared at an equal 4800 circuit executions, 5 seeds).**

| Optimizer | error at 4800 executions | runs reaching 1e-2 | iterations to 1e-2 | executions to 1e-2 |
|---|---|---|---|---|
| GD + parameter-shift | 1.2e-2 | 2/5 | 64 | 2064 |
| Adam + parameter-shift | 1.8e-3 | 5/5 | 51 | 1632 |
| SPSA | 3.9e-3 | 5/5 | 827 | 1654 |
| QNG (assumed metric cost 2P^2 = 512 per step) | 0.29 | 3/5 | 21 | 11424 |

Adam and SPSA need about the same number of circuit executions even though SPSA takes 16x more iterations: parameter-shift costs 2P per step, SPSA 2. QNG takes the fewest iterations (21) but is erratic (2 of 5 runs missed 1e-2) and, with an estimated metric cost, loses badly per circuit. Here the metric came from the simulated state, so its hardware cost is an assumption.

**Finite shots (n=8, p=2, SPSA 150 iterations).** The estimator std matches sqrt(var/S) (0.507 vs 0.476 at S=10; 0.0146 vs 0.0151 at S=10,000). Final ratio: 0.722 with 10 shots (start 0.700), 0.757 with 100, 0.778 with 1000, 0.775 with 10,000, and 0.775 with exact expectation. So 10 shots barely trains, 100 helps, and 1000 or more is indistinguishable from exact for this optimizer; the cost is 300,000 shots at 1000 per estimate. (The ceiling 0.775 vs the optimum 0.818 is SPSA's 150-iteration limit, not shot noise.)

**Noise (n=6, simulated depolarizing noise after every gate).** Noiseless: 0.754 / 0.858 / 0.888 for p = 1 / 2 / 3 (20 / 34 / 48 gates). At 3% per-gate noise they fall to 0.708 / 0.729 / 0.694: the deepest circuit becomes worse than p=1, and all approach the random-guess ratio 0.571. More layers help only while gate noise is small.

**Barren plateaus (random hardware-efficient circuits, 150 samples per point, n = 2..11).** Var[dC/dtheta] for a global cost with depth n falls by a factor 0.50 per added qubit (slope of ln Var = -0.68, close to -ln 2 = -0.69): 0.24 at n=2 to 4.9e-4 at n=11. A local cost at depth n decays more slowly (slope -0.45), and a local cost at fixed depth 2 does not decay (about 0.09 from n=5 to 11).

**Annealing-hardware limits.** Standard clique embedding of K_{4m} in a Chimera graph C_m needs chains of length m+1 and 4m(m+1) physical qubits (about n^2/4), so a 65-variable dense QUBO does not fit a 2048-qubit C16. In a toy chain experiment (8 logical variables, chains of 3, 20 SA runs per setting) chains always broke with strength 0, 3/20 broke at 0.5, none at >= 1.0. Quality (logical energy / optimum) was 0.81, 0.98, then drifted down 0.95 -> 0.93 as strength grew to 16, consistent with strong chains compressing the problem's energy scale (the same effect as the penalty weight in Track B); 20 runs on one instance cannot establish significance.

## Limitations

- Simulation only; no real device, no readout error, no coherent noise. The noise model is a single depolarizing channel.
- Graphs are small (6-12 nodes), so nothing here says how QAOA behaves at scale; classical methods solve all of them instantly.
- Only one Hamiltonian family (TFIM) for VQE; the slide's H2 coefficients were not used because the slide itself says to verify their source.
- A trained variational classifier / quantum kernel is in Track I; here the QNN is the slide's one-qubit example only.
- QNG hardware cost is an assumption (see above). No error mitigation was tried.
