"""Run xv6 under QEMU non-interactively: boot, type shell commands, collect output.
Usage:
  python bench.py smoke                 # boot, run schedbench once per policy on the short 'jobs' workload
  python bench.py bench [reps]          # all policies x workloads, results/bench.json
  python bench.py usertests [policy]    # run 'usertests -q' under a scheduling policy, results/usertests_<policy>.txt
The kernel is built by build.sh into $XV6_BUILD/src (one CPU: -smp 1, so the schedulers are compared on one core)."""
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("XV6_BUILD_WIN", r"C:\Users\pawan\AppData\Local\Temp\xv6build\src")
QEMU = r"C:\msys64\clang64\bin\qemu-system-riscv64.exe"
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


class XV6:
    def __init__(self, smp=1):
        env = {**os.environ, "PATH": r"C:\msys64\clang64\bin;" + os.environ["PATH"]}
        self.p = subprocess.Popen([QEMU, "-machine", "virt", "-bios", "none", "-kernel", "kernel/kernel", "-m", "128M", "-smp", str(smp), "-nographic",
                                   "-global", "virtio-mmio.force-legacy=false", "-drive", "file=fs.img,if=none,format=raw,id=x0",
                                   "-device", "virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0"],
                                  cwd=BUILD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        self.buf = ""
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            b = self.p.stdout.read(1)
            if not b:
                return
            with self.lock:
                self.buf += b.decode("latin-1")

    def wait_for(self, pattern, timeout):
        t = time.time()
        while time.time() - t < timeout:
            with self.lock:
                m = re.search(pattern, self.buf)
                if m:
                    out = self.buf[:m.end()]
                    self.buf = self.buf[m.end():]
                    return out
            time.sleep(0.05)
        raise TimeoutError(f"waiting for {pattern!r}; got tail: {self.buf[-400:]!r}")

    def run(self, cmd, until=r"\$ $", timeout=300):
        self.p.stdin.write((cmd + "\n").encode())
        self.p.stdin.flush()
        return self.wait_for(until, timeout)

    def close(self):
        self.p.kill()
        self.p.wait()


def parse(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("R "):
            kv = dict(re.findall(r"(\w+)=(-?\w+)", line))
            kv["_kind"] = line.split()[1]
            rows.append(kv)
    return rows


def boot():
    vm = XV6()
    vm.wait_for(r"init: starting sh\s*\$ ", 120)
    return vm


def smoke():
    vm = boot()
    for pol in (0, 1, 2):
        out = vm.run(f"schedbench {pol} jobs", until=r"R done\s*\$ ", timeout=300)
        print(pol, [r for r in parse(out) if r["_kind"] in ("begin", "jobs_summary")])
    vm.close()


def bench(reps=3):
    results = []
    for rep in range(reps):
        vm = boot()                                   # fresh boot per repetition
        vm.run("schedbench 0 jobs", until=r"R done\s*\$ ", timeout=600)      # warm-up (discarded): QEMU's JIT is slow at first
        order = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][rep % 3]                   # rotate policy order so no policy always runs first
        for wl in ("share", "mixed", "jobs"):
            for pol in order:
                out = vm.run(f"schedbench {pol} {wl}", until=r"R done\s*\$ ", timeout=600)
                results.append({"rep": rep, "policy": pol, "workload": wl, "rows": parse(out)})
                print(rep, wl, pol, flush=True)
        vm.close()
    json.dump(results, open(os.path.join(RES, "bench.json"), "w"), indent=1)


def usertests(policy=0):
    vm = boot()
    if policy:
        vm.run(f"setpolicy {policy}", timeout=30)
    t = time.time()
    out = vm.run("usertests -q", until=r"(ALL TESTS PASSED|SOME TESTS FAILED)\s*\$ ", timeout=1800)
    vm.close()
    open(os.path.join(RES, f"usertests_policy{policy}.txt"), "w").write(out)
    print(f"policy {policy}: {'PASSED' if 'ALL TESTS PASSED' in out else 'FAILED'} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "smoke": smoke()
    elif cmd == "bench": bench(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else: usertests(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
