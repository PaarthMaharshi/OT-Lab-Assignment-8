import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import passes as P  # noqa: E402
import programs as pr  # noqa: E402
import toyc  # noqa: E402


def run(src):
    return toyc.run_vm(toyc.compile_to_ir(src))[0]


def test_expression_precedence_and_c_semantics():
    assert run("print 2 + 3 * 4;") == [14]
    assert run("print (2 + 3) * 4 - 1;") == [19]
    assert run("print 7 / 2; print 0 - 7 / 2; print (0 - 7) / 2; print (0 - 7) % 3;") == [3, -3, -3, -1]   # truncation toward zero
    assert run("print 3 < 4; print 4 < 3; print 1 << 4;") == [1, 0, 16]


def test_control_flow():
    assert run("let i = 0; let s = 0; while (i < 5) { s = s + i; i = i + 1; } print s;") == [10]
    assert run("let x = 3; if (x > 2) { print 1; } else { print 2; }") == [1]


@pytest.mark.parametrize("name", list(pr.PROGRAMS))
def test_every_pass_and_pipeline_preserve_output(name):
    ir = toyc.compile_to_ir(pr.source(name, pr.SMALL[name]))
    ref = toyc.run_vm(ir)[0]
    for p in P.PASSES:
        assert toyc.run_vm(P.apply(ir, [p]))[0] == ref
    assert toyc.run_vm(P.fixpoint(ir))[0] == ref
    assert P.static_count(P.fixpoint(ir)) <= P.static_count(ir)


def test_individual_pass_effects():
    ir = toyc.compile_to_ir("let a = 2 + 3; print a * 8;")
    assert any(i == ("copy", "a", 5) for i in P.cprop(ir))                      # folded
    ir2 = toyc.compile_to_ir("let i = 0; while (i < 3) { print i * 8; i = i + 1; }")
    assert any(i[0] == "bin" and i[2] == "<<" and i[4] == 3 for i in P.strength(ir2))   # i*8 -> i<<3
    assert toyc.run_vm(P.strength(ir2))[0] == [0, 8, 16]
    ir = toyc.compile_to_ir("let x = 5; let y = 4; let a = x * y; let b = x * y; print a + b;")
    assert sum(1 for i in P.cse(ir) if i[0] == "bin" and i[2] == "*") == 1       # second x*y reused
    ir = toyc.compile_to_ir("let a = 1; let dead = a * 9; print a;")
    assert all(i[1] != "dead" for i in P.dce(ir) if i[0] in ("copy", "bin"))
    ir = toyc.compile_to_ir("let d = 0; if (d) { print 1; } else { print 2; }")
    opt = P.fixpoint(ir)
    assert ("print", 1) not in opt and toyc.run_vm(opt)[0] == [2]                # dead branch removed


def test_division_by_zero_is_not_removed_by_dce():
    ir = toyc.compile_to_ir("let z = 0; let bad = 5 / z; print 1;")
    assert any(i[0] == "bin" and i[2] == "/" for i in P.dce(ir))


def rand_expr(rng, vars_, d):
    if d == 0 or rng.random() < 0.25:
        return str(rng.choice([0, 1, 2, 3, 4, 8, 5])) if rng.random() < 0.5 else rng.choice(vars_)
    return f"({rand_expr(rng, vars_, d - 1)} {rng.choice(['+', '-', '*', '<', '=='])} {rand_expr(rng, vars_, d - 1)})"


def rand_prog(rng):
    vs = ["a", "b", "c", "d"]
    lines = [f"let {v} = {rng.randint(0, 9)};" for v in vs]
    def stmts(depth):
        out = []
        for _ in range(rng.randint(1, 4)):
            r = rng.random()
            if r < 0.55 or depth == 0:
                out.append(f"{rng.choice(vs)} = {rand_expr(rng, vs, 3)};")
            elif r < 0.75:
                out.append(f"if ({rand_expr(rng, vs, 2)}) {{ {' '.join(stmts(depth - 1))} }} else {{ {' '.join(stmts(depth - 1))} }}")
            else:
                out.append(f"print {rand_expr(rng, vs, 2)};")
        return out
    body = stmts(2)
    lines.append(f"let i = 0; while (i < {rng.randint(1, 4)}) {{ {' '.join(body)} i = i + 1; }}")
    lines += [f"print {v};" for v in vs]
    return "\n".join(lines)


def test_fuzz_random_programs_all_orders_preserve_behaviour():
    rng = random.Random(0)
    for _ in range(150):
        ir = toyc.compile_to_ir(rand_prog(rng))
        ref = toyc.run_vm(ir)[0]
        for name in P.PASSES:
            assert toyc.run_vm(P.apply(ir, [name]))[0] == ref
        order = rng.sample(list(P.PASSES), 4)
        assert toyc.run_vm(P.apply(ir, order))[0] == ref
        assert toyc.run_vm(P.fixpoint(ir))[0] == ref


def test_coloring_exact_never_worse_than_greedy_and_valid_lower_bound():
    for name in pr.PROGRAMS:
        ir = P.fixpoint(toyc.compile_to_ir(pr.source(name, 10)))
        nodes, edges = P.interference(ir)
        g = P.color_greedy(nodes, edges)
        e = P.color_exact(nodes, edges, g)
        assert 1 <= e <= g
