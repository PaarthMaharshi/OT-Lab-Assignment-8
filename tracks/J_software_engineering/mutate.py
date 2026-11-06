"""Mutation testing to measure what test-suite minimization costs in fault detection.

Generates simple mutants (operator swaps) on source lines that tests cover, runs the FULL suite once per mutant in an
isolated copy of the package, and records which tests fail (= which tests "kill" the mutant). Output: results/kills.json.
Usage: python mutate.py <package rel path> [n_mutants]
"""
import ast
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

from collect import RES, site_pkg

SWAPS_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
SWAPS_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add}


class Mutator(ast.NodeTransformer):
    def __init__(self, target_index, lines):
        self.i, self.k, self.lines, self.desc = target_index, -1, lines, None

    def visit_Compare(self, node):
        self.generic_visit(node)
        if type(node.ops[0]) in SWAPS_CMP and node.lineno in self.lines:
            self.k += 1
            if self.k == self.i:
                old = type(node.ops[0]).__name__
                node.ops[0] = SWAPS_CMP[type(node.ops[0])]()
                self.desc = f"line {node.lineno}: {old} -> {type(node.ops[0]).__name__}"
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in SWAPS_BIN and node.lineno in self.lines:
            self.k += 1
            if self.k == self.i:
                old = type(node.op).__name__
                node.op = SWAPS_BIN[type(node.op)]()
                self.desc = f"line {node.lineno}: {old} -> {type(node.op).__name__}"
        return node


def candidates(src, lines):
    """Number of mutation sites in this file on covered lines."""
    m = Mutator(-1, lines)
    m.visit(ast.parse(src))
    return m.k + 1


def run_mutant(args):
    rel, relfile, site_index, lines, workdir = args
    path = os.path.join(workdir, relfile)
    src = open(path, encoding="utf8").read()
    m = Mutator(site_index, lines)
    tree = m.visit(ast.parse(src))
    if m.desc is None:
        return None
    open(path, "w", encoding="utf8").write(ast.unparse(ast.fix_missing_locations(tree)))
    junit = os.path.join(workdir, "m.xml")
    if os.path.exists(junit):
        os.remove(junit)
    timed_out = False
    try:
        subprocess.run([sys.executable, "-m", "pytest", os.path.join(workdir, rel, "tests"), "-q", "-p", "no:cacheprovider", "--tb=no", "--timeout=20",
                        f"--junitxml={junit}"], env={**os.environ, "PYTHONPATH": workdir}, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        timed_out = True                                             # the whole run hung: the mutant is detected, but by no particular test
    finally:
        open(path, "w", encoding="utf8").write(src)                  # always restore the pristine file
    killed = []
    if os.path.exists(junit):
        for tc in ET.parse(junit).getroot().iter("testcase"):
            if tc.find("failure") is not None or tc.find("error") is not None:
                killed.append(f"{tc.get('classname')}.{tc.get('name')}")
    return {"file": relfile, "mutation": m.desc, "killed_by": killed, "run_timed_out": timed_out}


def main(rel, n):
    data = json.load(open(os.path.join(RES, "tests.json")))
    covered = {}
    for t in data["tests"].values():
        for l in t["lines"]:
            f, ln = l.rsplit(":", 1)
            covered.setdefault(f, set()).add(int(ln))
    target = site_pkg(rel)
    sites = []
    for f, lines in covered.items():
        full = os.path.join(target, f)
        for i in range(candidates(open(full, encoding="utf8").read(), lines)):
            sites.append((f, i, lines))
    random.Random(0).shuffle(sites)
    sites = sites[:n]
    tmp = tempfile.mkdtemp(prefix="mut_")
    jobs = []
    # one pristine copy of the package per worker slot, mutants rotate over them
    workers = min(6, os.cpu_count() or 2)
    roots = []
    pkg_top = rel.split("/")[0]
    import importlib.util
    top_dir = os.path.dirname(importlib.util.find_spec(pkg_top).origin)
    for w in range(workers):
        root = os.path.join(tmp, f"w{w}")
        shutil.copytree(top_dir, os.path.join(root, pkg_top), ignore=shutil.ignore_patterns("__pycache__"))
        roots.append(root)
    for j, (f, i, lines) in enumerate(sites):
        jobs.append((rel, os.path.join(rel, f).replace("\\", "/"), i, lines, roots[j % workers]))
    # mutants sharing a worker root must not run concurrently: run them in per-root sequential chunks
    by_root = {}
    for jb in jobs:
        by_root.setdefault(jb[4], []).append(jb)

    results = []
    with ProcessPoolExecutor(workers) as ex:
        for out in ex.map(run_chunk, list(by_root.values())):
            results += [r for r in out if r]
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(results, open(os.path.join(RES, "kills.json"), "w"), indent=1)
    print(f"{len(results)} mutants; killed by the full suite: {sum(bool(r['killed_by']) for r in results)}")


def run_chunk(chunk):
    return [run_mutant(j) for j in chunk]


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
