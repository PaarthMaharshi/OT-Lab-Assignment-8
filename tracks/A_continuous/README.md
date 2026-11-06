# Track A - Continuous Optimization

Everything is written from scratch in numpy (`optlib.py`) following the update rules in PPT 1, and checked against the slides' own worked examples.

Run: `python experiments.py` (about 2 minutes) -> `results/summary.json`, `results/fig1..fig5*.png`. Tests: `python -m pytest tests`.

## What is covered

| Requirement | Where | Evidence |
|---|---|---|
| ML as optimization (loss, regularization) | `problems.py` Logistic, MLP; exp4 | L2 sweep: weight norm 18.4 -> 2.3, test acc 0.96 -> 0.82 as lambda grows 0 -> 0.1 |
| Convex vs non-convex, local/global minima, saddle | exp1, `fig1` | double well: GD ends in -1.30 (global) or 1.13 (local) depending on start; saddle: GD from (1,0) stalls at the saddle, Newton goes to it |
| GD, SGD, mini-batch SGD | exp3, exp5 | equal 30-epoch gradient budget |
| Momentum, NAG, AdaGrad, RMSProp, Adam | exp3 | slide examples reproduced to 3-4 digits in tests |
| Newton, Gauss-Newton (and Levenberg-Marquardt) | exp3, exp7 | Newton 1 step on quadratic; GN 6 iterations on curve fit |
| Backpropagation (separate from the optimizer) | `MLP.grad`, exp4 | finite-difference check: max relative error 8e-10 |
| Curvature: Hessian-vector products, Fisher, low rank | exp6 | HVP error 3e-16 (exact) / 2e-10 (finite difference); rank-10 Hessian error 4.8%; Newton-CG, natural gradient |
| Convergence: convex, strongly convex, non-convex | exp2, `fig2` | measured f_k never exceeds the textbook bound (ratios 1.0 / 0.26 / 0.44) |
| Step-size and optimizer tuning | exp3 `fig3` | 25-point lr grid, tuned on tuning seeds, reported on different seeds |
| Noisy objectives | exp5 `fig5` | batch size 1..2000, constant vs decaying lr, 5 seeds |

## Fairness rules

Same gradient-evaluation budget per method; lr tuned on a grid (1e-4..1e2) then re-evaluated on different seeds; stochastic results are mean +- std over 5 seeds; full-batch Newton is reported by steps and wall time because one step costs a d x d solve.

## Findings (see `results/summary.json` for all numbers)

- **Curvature wins when it is cheap.** On logistic regression (d=20) Newton, Newton-CG and natural gradient all reach f-f* < 1e-10 in 4 steps; tuned GD needs 1938, Adam 3072. Newton-CG got there with 539 Hessian-vector products and never formed a Hessian. For generalised linear models natural gradient with the true Fisher plus lambda I reproduces Newton's iterates exactly (checked).
- **Newton is not a descent method.** It converged in 5 steps on Rosenbrock, but on the saddle function it converged to the saddle. Gradient methods also stall exactly on a saddle if started exactly on its stable manifold.
- **Empirical Fisher is not the Hessian.** It differs by 12% at the optimum and 95% far away; the true Fisher equals the Hessian exactly for logistic regression.
- **Adaptive methods and structure.** AdaGrad is far ahead on the kappa=100 quadratic and on logistic regression. That is largely because both problems are (nearly) diagonally scaled, which per-coordinate step sizes exploit; do not read it as "AdaGrad is best in general". On the non-diagonal Rosenbrock valley it does not reach 1e-6.
- **Noise.** With a constant step, smaller batches leave a higher error floor (bs=1: 1.79, bs=8: 0.12, bs=64: 0.012); decaying the step cuts it 27x (bs=1), 17x (bs=8) and 4x (bs=64). At bs=2000 (20 full steps) the error is not noise but unfinished optimization, and decaying the rate makes it worse.
- **MLP.** GD, SGD and Adam reach 95.5-96.5% test accuracy; the differences are within one standard deviation, so no claim that one is better.

## Limitations (honest)

- Learning rates are tuned per method per problem, which favours methods whose sweet spot is narrow; hyperparameters other than lr use textbook defaults.
- Diagonal-Fisher preconditioning uses an untuned damping constant (0.1); the other methods got a tuned lr.
- Logistic data is synthetic; MLP data is sklearn `make_moons`. No real-data claims.
- Wall-clock times come from one machine and are tiny (milliseconds); steps and gradient evaluations are the reliable comparison.
