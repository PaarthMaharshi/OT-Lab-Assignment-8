# Track G - xv6 Operating-System Optimization: Scheduling

**System:** MIT xv6-riscv (upstream commit `fa2ddcb`), QEMU 11 (`qemu-system-riscv64`, `-smp 1`) and a RISC-V GCC 16 cross compiler from MSYS2 on Windows.
**Submitted:** `xv6_scheduling.patch` (14 files, +442/-6; apply with `git apply` on that commit), `build.sh` (build), `bench.py` (boots xv6, types commands into the console, collects results), `analyze.py`. The upstream tree itself is not committed.

## What was changed (OS concept: CPU scheduling)
- **Feature 1 - lottery scheduler (proportional share).** Each process holds tickets (`settickets(n)`); each scheduling decision draws a random ticket (xorshift PRNG) among runnable processes.
- **Feature 2 - multilevel feedback queue (MLFQ).** 3 levels with time slices of 1, 2, 4 ticks; a process that uses its whole slice is demoted; every 50 ticks all processes are boosted to the top level (no starvation); a running process yields at the next tick if a higher-priority process is runnable.
- **Statistics system calls:** `setsched(policy)` switches the policy at run time (0 = the original round-robin loop, 1 = lottery, 2 = MLFQ); `getpinfo(struct pstat *)` reports per-process tickets, level, CPU ticks and times scheduled. Timer ticks are counted per process in `usertrap`.
- **Tick length 100 ms -> 10 ms** so the benchmarks run in seconds (same for all policies).
- Host portability only: `mkfs.c` uses binary-mode `open` and maps `bzero/bcopy/index` on Windows (no effect elsewhere).
User programs: `schedbench` (three workloads below) and `setpolicy`.

## Objectives and workloads (time unit: 10 ms tick; 3 boots, all three policies measured in every boot, order rotated, one discarded warm-up run per boot)
| Workload | Objective | Round robin | Lottery | MLFQ |
|---|---|---|---|---|
| **share**: 3 CPU-bound processes with 30/20/10 tickets for 6 s | CPU share proportional to tickets (ideal 0.50/0.33/0.17) | 0.33/0.33/0.33 (max error 0.168) | **0.49/0.34/0.17 (max error 0.008)** | 0.34/0.32/0.34 (0.170) |
| **mixed**: 2 CPU hogs + 1 interactive process (sleep 2 ticks, about 0.8 tick of work) | wake-up delay of the interactive process | **0.00 mean, max 1** | 2.0 mean, max 11 | **0.00-0.02 mean, max 1** |
| | interactive iterations in 6 s (max about 300) | 288-299 | 147-149 | 283-299 |
| **jobs**: 6 CPU jobs of 10-100 work units started together | mean turnaround (paired, same boot) | 1.00 | **1.05-1.10x (worse)** | **0.93-0.99x (better)** |
| | makespan | 140-202 | 141-203 | 141-207 |

(Absolute turnaround differs by boot: 90.8 / 93.6 / 132 ticks for round robin, because QEMU's speed varies between boots. That is why every comparison is made within one boot.)

**Findings**
- Each policy optimises a different objective; none dominates. Lottery is the only one that delivers proportional share (error < 1 percentage point), but it is the worst for interactive latency (random draws: mean 2-tick delay, worst 11 ticks) and for turnaround.
- MLFQ approximates shortest-job-first only weakly here (1-7% better mean turnaround than round robin): every job longer than 7 ticks ends up in the bottom queue and the 50-tick boost keeps resetting levels, so most of the time it behaves like round robin among long jobs.
- Makespan is the same for all policies (the CPU never idles): scheduling only changes *who waits*, not the total work.
- **First MLFQ version was flawed:** it preempted only when a slice expired, so a hog at the bottom level kept the CPU up to 4 ticks after the interactive process woke (mean delay 0.37-0.51 ticks, max 3-4, 237-252 iterations). Adding "yield if a higher-priority process is runnable" brought it to round-robin's responsiveness while keeping hogs demoted.
- Round robin with a 1-tick quantum is already excellent for this tiny interactive workload: with only 3 processes, waiting one quantum is short.

## Correctness
- Every policy runs xv6's own regression suite `usertests -q` (see `results/usertests_policy{0,1,2}.txt`): **ALL TESTS PASSED under all three policies** (round robin 122 s, lottery 97 s, MLFQ 95 s of wall time; run on the final kernel, after the MLFQ fix).
- The patched kernel boots under every policy, and the benchmark programs exit cleanly; results come from 27 measured runs (3 boots x 3 workloads x 3 policies).

## Measurement caveats
- QEMU (TCG emulation) speed varies by up to about 40% between boots and the first run after boot is slow (JIT warm-up): a first benchmark that let round robin always run first made it look 60% worse. Fixed by a discarded warm-up run, rotating the policy order, and comparing within a boot.
- Per-process CPU ticks count only ticks that land in user mode, so they under-count total CPU time; the CPU-share results use loop iterations instead.
- Work is calibrated as a fixed iteration count (a self-calibrating version varied 3.4-5.3 M iterations/tick between runs and was rejected).
- One CPU (`-smp 1`); the lottery and MLFQ pickers are written for one CPU and are not tuned for SMP.
