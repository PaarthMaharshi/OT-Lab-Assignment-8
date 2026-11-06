# Track F - Compiler Optimization

A small compiler for a toy integer language, written for this assignment: lexer, recursive-descent parser, three-address IR, four optimization passes, an IR interpreter (VM) and a C backend compiled with `gcc -O0` (so GCC does not repeat our optimizations).
Run: `python experiments.py` (about 2 minutes). Tests: `python -m pytest tests` (12 tests including 150 random programs).

| Requirement | Where | Evidence |
|---|---|---|
| Lexer, parser, IR, code generator | `toyc.py` | precedence, C-style truncating division, control flow tested |
| >= 4 optimizations | `passes.py`: `cprop` (constant/copy propagation, folding, branch folding), `strength` (x*2^k -> x<<k, x+0, x*1, x*0, x-x), `cse` (local value numbering), `dce` (liveness + unreachable blocks) | each pass tested on a targeted example |
| Compiler decision as an optimization problem | pass ordering (greedy vs exhaustive over 4! = 24 orders), register allocation (greedy colouring vs exact ILP) | tables below |
| Same output as unoptimized | asserted for every program and variant, plus fuzzing | 150 random programs x every pass x random orders x fixpoint, all identical |
| Measure instruction count, run time, code size, compile time | `experiments.py` | `results/summary.json` |

## Results (6 programs, `gcc -O0`, runtime = best of 7 after a warm-up)

Timing noise matters: building and timing the *same* code twice (A/A control) gave ratios 0.97-1.15. A gain is only believable if clearly outside that band.

| Program | static instr. | dynamic instr. | runtime speedup (A/A noise) | exe size |
|---|---|---|---|---|
| const_heavy | -22.7% | -8.3% | 1.21x (0.97) | -0.4% |
| cse_heavy | -27.6% | -33.3% | 1.51x (1.15) | -0.02% |
| strength_heavy | -5.3% | -6.2% | 0.99x (1.00) | -0.01% |
| dead_code | -38.9% | -46.7% | 1.28x (0.97) | -0.02% |
| mixed (resolvable branch + CSE) | -11.5% | -16.7% | 0.99x (1.00) | -0.4% |
| control_primes (nothing to optimize) | 0% | 0% | 1.01x (1.02) | 0% |

What this shows, honestly:
- Dead-code and CSE gains are real (outside the noise); constant folding is borderline (1.21x vs noise 0.97-1.15).
- **Fewer instructions did not always mean faster:** `strength_heavy` and `mixed` cut dynamic instructions by 6% and 17% with no measurable speedup. At `-O0` the CPU hides a multiply-versus-shift difference, and removing cheap instructions does not shorten the critical path. Instruction count is a proxy, not the goal.
- The control program is unchanged and timed at 1.01x: the method does not invent speedups.
- Executable size barely moves (the C runtime dominates); the instruction counts are the meaningful size measure. Compile time for all passes is milliseconds.
- The dynamic counts come from the VM on a small input (2000 iterations, 300 for the primes control).

## Pass ordering as a search problem

Objective: fewest IR instructions after applying each of the 4 passes once, in some order. Exhaustive search tries all 24 orders (96 pass runs); greedy applies, at each step, the unused pass that shrinks the code most (10 pass runs).

| Program | unoptimized | best of 24 | worst of 24 | greedy | greedy = best? | run to fixpoint |
|---|---|---|---|---|---|---|
| const_heavy | 22 | 17 | 22 | 17 | yes | 17 |
| cse_heavy | 29 | 24 | 29 | 27 | no | 21 |
| strength_heavy | 19 | 18 | 19 | 19 | no | 18 |
| dead_code | 18 | 11 | 11 | 11 | yes | 11 |
| mixed | 26 | 23 | 24 | 24 | no | 23 |
| control_primes | 24 | 24 | 24 | 24 | yes | 24 |

Greedy finds the optimum order in 3 of 6 programs at about a tenth of the work; it misses when passes enable each other (a later `cprop` exposes work for `cse` or `dce`). Running the passes to a fixpoint beats every single-application order on `cse_heavy` (21 vs 24), so *repetition* matters more than *order* there. Pass order mattered most for `const_heavy` (22 vs 17) and not at all for `dead_code` and the control.

## Register allocation as graph colouring

Interference graph from liveness; greedy (Welsh-Powell) versus the exact chromatic number by ILP. Greedy matched the exact optimum on every graph (6/6 programs, before and after optimization; 3 to 7 registers). Optimization reduced the variable count (e.g. 24 -> 16 for `cse_heavy`) but cut the register need only for `mixed` (7 -> 6), because register need is set by the loop's peak liveness, not the variable count.

## Limitations
- Local (basic-block) propagation and CSE only; no loop-invariant code motion, inlining or global value numbering. No functions in the toy language.
- Programs are small and hand-written for the passes to matter; real-compiler gains are not claimed. Single machine, single compiler (gcc -O0).
- The register experiment counts colours only; no spilling or machine-specific costs.
