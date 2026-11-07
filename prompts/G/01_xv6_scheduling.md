# Prompt log entry - Track G (xv6 scheduling)

- **AI tool + model version:** Claude Code. Kernel patch and benchmark programs started with Claude Sonnet 5.5 (`claude-sonnet-5-5`); build, measurement, MLFQ fix and write-up with Claude Opus 5.5 (`claude-opus-5-5`), 2026-11-07.
- **Prompt:** "just do everything else without asking me" / "do it quick please" (Track G from the assignment PDF).
- **Output:** `xv6_scheduling.patch` (lottery + MLFQ schedulers, `settickets`, `setsched`, `getpinfo`, 10 ms tick, `schedbench`, `setpolicy`), `build.sh`, `bench.py`, `analyze.py`. Commit: [cf698c6](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/cf698c6).
- **Verification:** xv6 `usertests -q` passes under round robin, lottery and MLFQ on the final kernel (`results/usertests_policy*.txt`); the benchmark programs exit cleanly under every policy; 27 measured runs (3 boots x 3 workloads x 3 policies).

## Environment work (not code, but necessary)
WSL was not installed and installing it would change system settings, so QEMU and the RISC-V GCC were installed into MSYS2's separate `clang64` environment instead (the `ucrt64` versions would have forced an upgrade of the existing GCC libraries). The upstream xv6-riscv source was cloned from `github.com/mit-pdos/xv6-riscv`; only the patch is committed.

## Issues found (feed docs/failure_log.md)
1. **Wrong line endings.** The Windows clone converted files to CRLF, so the first patch script failed and the Makefile would have broken. Fixed by re-checking out with LF endings.
2. **Build environment failures, one after another:** the Makefile could not auto-detect the toolchain prefix; the project path contains a space, which breaks `make`; the native compiler wrote temp files to `C:\WINDOWS` under `make`; `ucrt64` DLLs on PATH crashed the cross compiler (error 127). Fixed in `build.sh`: build in a space-free copy, explicit `TOOLPREFIX`, `TMPDIR` plus `-pipe`, and a minimal PATH.
3. **API mismatch with this xv6 version:** `copyout` takes five arguments here (not four), and the user-level sleep call is named `pause`. Both caught by the compiler.
4. **Host portability:** `mkfs` used BSD functions (`bzero`, `bcopy`, `index`) and text-mode `open`, which would corrupt the disk image on Windows; small `#ifdef _WIN32` shims and `O_BINARY` added.
5. **Biased benchmark design (two flaws).** Self-calibrating the work unit gave 3.4-5.3 M iterations per tick between runs (QEMU JIT warm-up), and round robin always ran first after boot, making it look about 60% worse. Fixed with a fixed work unit, a discarded warm-up run per boot, and rotating the policy order.
6. **MLFQ design flaw found by measurement.** The first MLFQ preempted only when a time slice expired, so a CPU hog at the bottom level kept the CPU up to 4 ticks after an interactive process woke (mean delay 0.37-0.51 ticks, max 3-4). Added "yield if a higher-priority process is runnable"; delay fell to 0.00-0.02 ticks. The usertests run that had started on the old kernel was stopped and redone on the fixed one.
7. **Escape mangling** of `\n` and `\\` in shell heredocs broke the Makefile edit and C string literals; files written with the editor tool instead.

## Chat export / screenshot
_add export_
