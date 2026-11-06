"""Optimization passes on the three-address IR and the liveness analysis they share.

Four passes (all local to basic blocks except DCE, which uses global liveness):
  cprop    constant/copy propagation + constant folding + branch folding
  strength strength reduction and algebraic identities (x*2^k -> x<<k, x+0 -> x, x*1 -> x, x*0 -> 0)
  cse      common sub-expression elimination (local value numbering)
  dce      dead-code elimination (liveness) + unreachable-block removal
"""
from __future__ import annotations

from toyc import ev, uses

COMMUTATIVE = {"+", "*", "==", "!="}


def _subst(ins, env):
    f = lambda x: env.get(x, x) if isinstance(x, str) else x
    op = ins[0]
    if op == "copy": return ("copy", ins[1], f(ins[2]))
    if op == "bin": return ("bin", ins[1], ins[2], f(ins[3]), f(ins[4]))
    if op == "print": return ("print", f(ins[1]))
    if op == "br": return ("br", f(ins[1]), ins[2], ins[3])
    return ins


def _kill(env, d):
    env.pop(d, None)
    for k in [k for k, v in env.items() if v == d]:
        del env[k]


def cprop(ir):
    out, env = [], {}
    for ins in ir:
        if ins[0] == "label":
            env = {}
            out.append(ins)
            continue
        ins = _subst(ins, env)
        if ins[0] == "bin" and isinstance(ins[3], int) and isinstance(ins[4], int):
            if not (ins[2] in ("/", "%") and ins[4] == 0):
                ins = ("copy", ins[1], ev(ins[2], ins[3], ins[4]))                # constant folding
        if ins[0] == "br" and isinstance(ins[1], int):
            ins = ("jmp", ins[2] if ins[1] != 0 else ins[3])                      # branch folding
        if ins[0] in ("copy", "bin"):
            _kill(env, ins[1])
            if ins[0] == "copy" and ins[2] != ins[1]:
                env[ins[1]] = ins[2]
        out.append(ins)
    return out


def strength(ir):
    out = []
    for ins in ir:
        if ins[0] == "bin":
            _, d, op, a, b = ins
            if op == "*":
                for x, c in ((a, b), (b, a)):
                    if isinstance(c, int):
                        if c == 0: ins = ("copy", d, 0); break
                        if c == 1: ins = ("copy", d, x); break
                        if c > 1 and c & (c - 1) == 0: ins = ("bin", d, "<<", x, c.bit_length() - 1); break
            elif op == "+":
                if b == 0 and isinstance(b, int): ins = ("copy", d, a)
                elif a == 0 and isinstance(a, int): ins = ("copy", d, b)
            elif op == "-":
                if isinstance(b, int) and b == 0: ins = ("copy", d, a)
                elif a == b and isinstance(a, str): ins = ("copy", d, 0)
            elif op == "/" and isinstance(b, int) and b == 1:
                ins = ("copy", d, a)
        out.append(ins)
    return out


def cse(ir):
    out, avail = [], {}                       # (op,a,b) -> variable holding it
    for ins in ir:
        if ins[0] == "label":
            avail = {}
            out.append(ins)
            continue
        if ins[0] == "bin":
            _, d, op, a, b = ins
            key = (op,) + tuple(sorted((a, b), key=str)) if op in COMMUTATIVE else (op, a, b)
            if key in avail and avail[key] != d:
                ins = ("copy", d, avail[key])
        if ins[0] in ("copy", "bin"):
            d = ins[1]
            for k in [k for k, v in avail.items() if v == d or d in k[1:]]:
                del avail[k]
            if ins[0] == "bin" and d not in (ins[3], ins[4]):
                _, d, op, a, b = ins
                key = (op,) + tuple(sorted((a, b), key=str)) if op in COMMUTATIVE else (op, a, b)
                avail[key] = d
        out.append(ins)
    return out


# ------------------------------------------------------------------ CFG / liveness
def blocks(ir):
    """Split into basic blocks: list of (start, end) index ranges."""
    starts = {0}
    for i, ins in enumerate(ir):
        if ins[0] == "label": starts.add(i)
        if ins[0] in ("jmp", "br") and i + 1 < len(ir): starts.add(i + 1)
    s = sorted(starts)
    return [(a, b) for a, b in zip(s, s[1:] + [len(ir)])]


def successors(ir, bl):
    lab = {ins[1]: bi for bi, (a, b) in enumerate(bl) for ins in [ir[a]] if ins[0] == "label"}
    succ = []
    for bi, (a, b) in enumerate(bl):
        last = ir[b - 1]
        if last[0] == "jmp": succ.append([lab[last[1]]])
        elif last[0] == "br": succ.append([lab[last[2]], lab[last[3]]])
        else: succ.append([bi + 1] if bi + 1 < len(bl) else [])
    return succ


def liveness(ir):
    """Return (blocks, live_out per block). Variables are live if a later instruction may read them."""
    bl = blocks(ir)
    succ = successors(ir, bl)
    use_b, def_b = [], []
    for a, b in bl:
        u, d = set(), set()
        for ins in ir[a:b]:
            u |= {x for x in uses(ins) if x not in d}
            if ins[0] in ("copy", "bin"): d.add(ins[1])
        use_b.append(u); def_b.append(d)
    live_in = [set() for _ in bl]; live_out = [set() for _ in bl]
    changed = True
    while changed:
        changed = False
        for bi in reversed(range(len(bl))):
            lo = set().union(*[live_in[s] for s in succ[bi]]) if succ[bi] else set()
            li = use_b[bi] | (lo - def_b[bi])
            if lo != live_out[bi] or li != live_in[bi]:
                live_out[bi], live_in[bi], changed = lo, li, True
    return bl, live_out


def _removable(ins):
    if ins[0] == "copy": return True
    if ins[0] == "bin": return not (ins[2] in ("/", "%") and not (isinstance(ins[4], int) and ins[4] != 0))
    return False


def dce(ir):
    # 1. unreachable blocks
    bl = blocks(ir); succ = successors(ir, bl)
    seen, stack = set(), [0]
    while stack:
        b = stack.pop()
        if b in seen: continue
        seen.add(b); stack += succ[b]
    ir = [ins for bi, (a, b) in enumerate(bl) if bi in seen for ins in ir[a:b]]
    # 2. dead assignments, repeated to a fixpoint
    while True:
        bl, live_out = liveness(ir)
        keep = []
        removed = False
        for bi, (a, b) in enumerate(bl):
            live = set(live_out[bi]); block = []
            for ins in reversed(ir[a:b]):
                if ins[0] in ("copy", "bin") and ins[1] not in live and _removable(ins):
                    removed = True
                    continue
                if ins[0] in ("copy", "bin"): live.discard(ins[1])
                live |= set(uses(ins))
                block.append(ins)
            keep += reversed(block)
        ir = keep
        if not removed:
            return ir


PASSES = {"cprop": cprop, "strength": strength, "cse": cse, "dce": dce}


def apply(ir, order):
    for name in order:
        ir = PASSES[name](ir)
    return ir


def fixpoint(ir, order=("cprop", "strength", "cse", "cprop", "dce"), limit=10):
    for _ in range(limit):
        new = apply(ir, order)
        if new == ir: break
        ir = new
    return ir


def static_count(ir):
    return sum(1 for i in ir if i[0] != "label")


# --------------------------------------------------------------- register allocation
def interference(ir):
    """Edges between variables that are live at the same time."""
    bl, live_out = liveness(ir)
    edges, nodes = set(), set()
    for bi, (a, b) in enumerate(bl):
        live = set(live_out[bi])
        for ins in reversed(ir[a:b]):
            if ins[0] in ("copy", "bin"):
                d = ins[1]; nodes.add(d)
                for x in live:
                    if x != d and not (ins[0] == "copy" and ins[2] == x):
                        edges.add(tuple(sorted((d, x))))
                live.discard(d)
            live |= set(uses(ins)); nodes |= set(uses(ins))
    return sorted(nodes), sorted(edges)


def color_greedy(nodes, edges):
    """Welsh-Powell: colour by decreasing degree with the smallest free colour."""
    adj = {n: set() for n in nodes}
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    col = {}
    for n in sorted(nodes, key=lambda n: -len(adj[n])):
        used = {col[m] for m in adj[n] if m in col}
        col[n] = next(c for c in range(len(nodes) + 1) if c not in used)
    return max(col.values()) + 1 if col else 0


def color_exact(nodes, edges, ub):
    """Chromatic number by ILP (binary x[v,c]); ub = greedy colour count."""
    from ortools.linear_solver import pywraplp
    if not nodes: return 0
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = {(v, c): s.BoolVar("") for v in nodes for c in range(ub)}
    u = [s.BoolVar("") for _ in range(ub)]
    for v in nodes: s.Add(sum(x[v, c] for c in range(ub)) == 1)
    for a, b in edges:
        for c in range(ub): s.Add(x[a, c] + x[b, c] <= u[c])
    for v in nodes:
        for c in range(ub): s.Add(x[v, c] <= u[c])
    s.Minimize(sum(u))
    s.Solve()
    return round(s.Objective().Value())
