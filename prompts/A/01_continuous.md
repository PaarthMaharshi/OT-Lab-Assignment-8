# Prompt log entry - Track A continuous optimization

- **Track / task:** A - optimizers, landscapes, backprop, curvature, convergence, noise
- **Date:** 2026-11-06
- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`)

## 1. Original prompt
"Start Track A" (assignment PDF Track A + the Chapter 6 PPT 1/4 content already read).

## 2. AI response (relevant excerpt)
`optlib.py` (GD, momentum, NAG, AdaGrad, RMSProp, Adam, Newton, Newton-CG, Gauss-Newton), `problems.py`, `tests/test_optlib.py`, `experiments.py`.

## 3. Revised prompt
None typed by me; the AI revised its own code after reading results (section 6).

## 4. Resulting code change
Commit: _see git log_. Files: `tracks/A_continuous/**`.

## 5. Verification note
- 25 pytest tests. Expected values are taken from the lecture slides' worked examples (GD 0.6/1.08/1.464, momentum 4/2.3, NAG 4/2.48, Adam 2.9, AdaGrad 2/1.445, RMSProp 2.684, Newton 1.5), from finite-difference gradient/HVP checks, and from textbook convergence bounds.
- Backprop checked against central finite differences (max relative error ~1e-9).
- Newton-CG, natural gradient and Newton compared on the same problem; natural gradient reproduces Newton iterates exactly (GLM property).
- Learning rates tuned on tuning seeds, reported on different seeds.

## 6. Issues found (feed docs/failure_log.md)
1. **Wrong test premise.** The AI asserted "empirical Fisher is closer to the Hessian near the optimum than at w=0". At w=0 every prediction is 0.5, so they coincide exactly (distance 0) and the test failed. Detected by the failing test; fixed by testing at 3 x w*. Lesson: check a claim at the chosen point before encoding it as a test.
2. **Benchmark artefact.** With the start x0 = all ones, AdaGrad's first step lr*g/|g| lands exactly on the optimum of the quadratic, so AdaGrad "converged in 1 step". Detected by reading the results table (final value exactly 0). Fix: random start point. Lesson: a too-good result is a bug until explained.
3. **Truncated tuning grid.** Full-batch GD's best learning rate sat on the edge of the grid (10), so tuning might be cut off. Fix: grid widened to 1e-4..1e2.
4. **Plain bug.** `len(slice)` raised TypeError in the logistic gradient; caught by the first test run.
5. **Unreadable plot.** Diverged runs (values ~1e192) squashed the learning-rate figure; y-axis clipped.
6. **Dead code** left in `experiments.py` (an `if False` expression and a `pass` branch); found on review and removed before running.

## 7. Chat export / screenshot
_add export_
