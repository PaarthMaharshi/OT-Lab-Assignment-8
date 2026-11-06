"""Collect, for every test of a target open-source package: its runtime (plain run) and the source lines it covers
(coverage.py with one dynamic context per test). Output: results/tests.json.

Usage: python collect.py <package_dir_with_tests_relative_to_site_packages> e.g. networkx/algorithms/shortest_paths
"""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def site_pkg(rel):
    import importlib.util
    spec = importlib.util.find_spec(rel.split("/")[0])
    return os.path.join(os.path.dirname(os.path.dirname(spec.origin)), rel)


def nodeid_to_key(nodeid):
    p = nodeid.split("|")[0]
    return p.replace("/", ".").replace(".py::", ".").replace("::", ".")


def main(rel):
    target = site_pkg(rel)
    tests_dir = os.path.join(target, "tests")
    junit = os.path.join(RES, "plain.xml")
    subprocess.run([sys.executable, "-m", "pytest", tests_dir, "-q", "-p", "no:cacheprovider", "--tb=no", f"--junitxml={junit}"], check=False,
                   capture_output=True)
    times = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        times[f"{tc.get('classname')}.{tc.get('name')}"] = float(tc.get("time"))
    cov_file = os.path.join(RES, ".coverage")
    if os.path.exists(cov_file):
        os.remove(cov_file)
    pkg = rel.replace("/", ".")
    subprocess.run([sys.executable, "-m", "pytest", tests_dir, "-q", "-p", "no:cacheprovider", "--tb=no", f"--cov={pkg}", "--cov-context=test",
                    "--cov-report=", "-o", f"cache_dir={RES}"], check=False, capture_output=True, env={**os.environ, "COVERAGE_FILE": cov_file})
    import coverage
    data = coverage.CoverageData(basename=cov_file)
    data.read()
    lines_by_test = {}
    for f in data.measured_files():
        if "/tests/" in f.replace("\\", "/") or "test_" in os.path.basename(f):
            continue
        for ln, ctxs in data.contexts_by_lineno(f).items():
            for c in ctxs:
                if c:
                    lines_by_test.setdefault(nodeid_to_key(c), set()).add(f"{os.path.relpath(f, target).replace(chr(92), '/')}:{ln}")
    tests = {}
    for k, lines in lines_by_test.items():
        if k in times:
            tests[k] = {"time": times[k], "lines": sorted(lines)}
    src_lines = 0
    for root, _, files in os.walk(target):
        if "tests" in root:
            continue
        for fn in files:
            if fn.endswith(".py"):
                src_lines += sum(1 for _ in open(os.path.join(root, fn), encoding="utf8"))
    json.dump({"package": rel, "source_lines_of_code_files": src_lines, "tests_with_coverage": len(tests), "tests_total": len(times), "tests": tests},
              open(os.path.join(RES, "tests.json"), "w"))
    print(f"{len(tests)} tests with coverage of {len(times)}; {src_lines} source lines in the package")


if __name__ == "__main__":
    main(sys.argv[1])
