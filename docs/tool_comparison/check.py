"""Automatic checker for the tool-comparison task. Usage: python check.py <solution_file.py>
Checks correctness against brute force (n <= 8, 60 random instances incl. asymmetric and n = 1, 2), tour validity,
and runtime at n = 13 and 15. Prints a summary line for docs/tool_comparison.md."""
import importlib.util
import itertools
import random
import sys
import time


def load(path):
    spec = importlib.util.spec_from_file_location("sol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.held_karp


def brute(d):
    n = len(d)
    if n == 1:
        return 0
    return min(sum(d[a][b] for a, b in zip((0,) + p, p + (0,))) for p in itertools.permutations(range(1, n)))


def tour_cost(d, t):
    return sum(d[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))


def main(path):
    hk = load(path)
    rng = random.Random(0)
    passed = failed = 0
    errors = []
    for k in range(60):
        n = [1, 2, 3][k] if k < 3 else rng.randint(4, 8)
        sym = k % 2 == 0
        d = [[0 if i == j else rng.randint(1, 100) for j in range(n)] for i in range(n)]
        if sym:
            for i in range(n):
                for j in range(i):
                    d[i][j] = d[j][i]
        try:
            cost, tour = hk(d)
            ok = (abs(cost - brute(d)) < 1e-9 and sorted(tour) == list(range(n)) and tour[0] == 0 and abs(tour_cost(d, tour) - cost) < 1e-9)
        except Exception as e:                       # noqa: BLE001
            ok = False
            errors.append(f"n={n}: {type(e).__name__}: {e}")
        passed += ok
        failed += not ok
        if not ok and len(errors) < 3:
            errors.append(f"n={n} sym={sym}: wrong or invalid result")
    times = {}
    for n in (13, 15):
        d = [[0 if i == j else rng.randint(1, 100) for j in range(n)] for i in range(n)]
        t = time.perf_counter()
        try:
            hk(d)
            times[n] = round(time.perf_counter() - t, 2)
        except Exception as e:                       # noqa: BLE001
            times[n] = f"error {type(e).__name__}"
    print(f"correct {passed}/60, failed {failed}; runtime n=13: {times[13]} s, n=15: {times[15]} s")
    for e in errors[:3]:
        print("  ", e)


if __name__ == "__main__":
    main(sys.argv[1])
