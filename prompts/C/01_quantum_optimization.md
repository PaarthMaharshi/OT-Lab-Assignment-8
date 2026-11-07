# Prompt log entry - Track C quantum optimization

- **Track / task:** C - numpy quantum simulator, VQE, QAOA, parameter-shift, SPSA, QNG, shots, noise, barren plateaus, embedding limits
- **Date:** 2026-11-06
- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`)

## 1. Original prompt
"Start Track C", then "just do everything else without asking me".

## 2. AI response (relevant excerpt)
`qsim.py`, `qopt.py`, `tests/test_quantum.py`, `experiments.py`. PennyLane was installed only to cross-check the simulator.

## 3. Revised prompt
None typed by me.

## 4. Resulting code change
Commit: [2528313](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/2528313). Files: `tracks/C_quantum_optimization/**`.

## 5. Verification note
- 14 tests: Bell state, Hadamard interference, Born rule and 1/sqrt(shots), slide VQE (0.740, 1.077), slide QNN (0.825), slide QAOA optima (3.237, 3.856), 12-parameter VQE reaching -4, parameter-shift vs finite differences (including shared QAOA parameters), gate-level vs fast QAOA, density matrix vs statevector, depolarizing channel analytic values, PennyLane cross-check.

## 6. Issues found (feed docs/failure_log.md)
1. **Unfair optimizer comparison (first version).** QNG "failed" (final error 5.4) because it was untuned (regulariser far too small for a near-singular metric) while other optimizers also used untuned defaults. Detected by the implausible number. Fix: tune every optimizer's hyperparameters on separate seeds and compare at an equal circuit-execution budget. QNG is still the worst per circuit, but now for a defensible reason.
2. **Experiment measured the wrong thing.** The shot-noise study used COBYLA, which terminated after about 40 evaluations on the noisy objective, so the results reflected early stopping, not shot noise (10,000 shots looked no better than 1,000). Replaced by SPSA with a fixed iteration count.
3. **Missing baseline.** "Best cut in 1000 QAOA shots = 100%" looked impressive, but best-of-1000 uniformly random bitstrings is also 100% for n <= 12. Baseline added and the claim withdrawn.
4. **Results lost to a serialization bug.** The first full run crashed writing JSON (`numpy int64`), discarding about 3 minutes of results; fixed with a conversion fallback.
5. **Overstated trend.** The chain-strength experiment (20 runs, one instance) shows a small downward drift in quality that I first wrote as a finding; it is reported only as "consistent with", with the significance caveat.

## 7. Chat export / screenshot
_add export_
