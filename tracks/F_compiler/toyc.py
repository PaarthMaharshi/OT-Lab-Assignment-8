"""A small compiler for a toy integer language: lexer -> parser -> three-address IR -> (passes) -> C / VM.

Language:  let x = e;   x = e;   while (e) { ... }   if (e) { ... } else { ... }   print e;
Expressions: + - * / % << comparisons (< > <= >= == !=, result 0/1), unary minus, parentheses.
IR (list of tuples; an operand is an int constant or a variable name):
    ("label", L)  ("jmp", L)  ("br", cond, Ltrue, Lfalse)  ("copy", dst, src)
    ("bin", dst, op, a, b)  ("print", a)
Semantics are C-like (division truncates toward zero) so the C backend and the VM agree.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(<<|<=|>=|==|!=|[-+*/%<>=(){};]))")
KEYWORDS = {"let", "while", "if", "else", "print"}


def lex(src):
    pos, out = 0, []
    src = re.sub(r"#.*", "", src)
    while pos < len(src):
        if src[pos:].strip() == "":
            break
        m = TOKEN.match(src, pos)
        if not m:
            raise SyntaxError(f"bad character at {pos}: {src[pos:pos + 10]!r}")
        pos = m.end()
        num, ident, sym = m.groups()
        out.append(("num", int(num)) if num else ("id", ident) if ident else ("sym", sym))
    return out


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self): return self.t[self.i] if self.i < len(self.t) else ("eof", None)

    def eat(self, kind=None, val=None):
        tok = self.peek()
        if (kind and tok[0] != kind) or (val is not None and tok[1] != val):
            raise SyntaxError(f"expected {val or kind}, got {tok}")
        self.i += 1
        return tok

    def program(self):
        out = []
        while self.peek()[0] != "eof":
            out.append(self.stmt())
        return out

    def block(self):
        self.eat("sym", "{")
        out = []
        while self.peek() != ("sym", "}"):
            out.append(self.stmt())
        self.eat("sym", "}")
        return out

    def stmt(self):
        k, v = self.peek()
        if (k, v) == ("id", "let"):
            self.eat(); name = self.eat("id")[1]; self.eat("sym", "="); e = self.expr(); self.eat("sym", ";")
            return ("let", name, e)
        if (k, v) == ("id", "while"):
            self.eat(); self.eat("sym", "("); c = self.expr(); self.eat("sym", ")")
            return ("while", c, self.block())
        if (k, v) == ("id", "if"):
            self.eat(); self.eat("sym", "("); c = self.expr(); self.eat("sym", ")")
            a = self.block(); b = []
            if self.peek() == ("id", "else"):
                self.eat(); b = self.block()
            return ("if", c, a, b)
        if (k, v) == ("id", "print"):
            self.eat(); e = self.expr(); self.eat("sym", ";")
            return ("print", e)
        name = self.eat("id")[1]; self.eat("sym", "="); e = self.expr(); self.eat("sym", ";")
        return ("assign", name, e)

    LEVELS = [("<", ">", "<=", ">=", "==", "!="), ("<<",), ("+", "-"), ("*", "/", "%")]

    def expr(self, lvl=0):
        if lvl == len(self.LEVELS):
            return self.unary()
        left = self.expr(lvl + 1)
        while self.peek()[0] == "sym" and self.peek()[1] in self.LEVELS[lvl]:
            op = self.eat()[1]
            left = ("bin", op, left, self.expr(lvl + 1))
        return left

    def unary(self):
        if self.peek() == ("sym", "-"):
            self.eat()
            return ("bin", "-", ("num", 0), self.unary())
        k, v = self.peek()
        if k == "num": self.eat(); return ("num", v)
        if k == "id" and v not in KEYWORDS: self.eat(); return ("var", v)
        self.eat("sym", "(")
        e = self.expr(); self.eat("sym", ")")
        return e


def parse(src):
    return Parser(lex(src)).program()


# ---------------------------------------------------------------- IR generation
class Gen:
    def __init__(self):
        self.ir, self.nt, self.nl = [], 0, 0

    def tmp(self): self.nt += 1; return f"t{self.nt}"
    def lab(self): self.nl += 1; return f"L{self.nl}"

    def expr(self, e):
        if e[0] == "num": return e[1]
        if e[0] == "var": return e[1]
        a = self.expr(e[2]); b = self.expr(e[3]); d = self.tmp()
        self.ir.append(("bin", d, e[1], a, b))
        return d

    def stmts(self, ss):
        for s in ss:
            k = s[0]
            if k in ("let", "assign"):
                self.ir.append(("copy", s[1], self.expr(s[2])))
            elif k == "print":
                self.ir.append(("print", self.expr(s[1])))
            elif k == "while":
                lc, lb, le = self.lab(), self.lab(), self.lab()
                self.ir.append(("label", lc))
                c = self.expr(s[1])
                self.ir += [("br", c, lb, le), ("label", lb)]
                self.stmts(s[2])
                self.ir += [("jmp", lc), ("label", le)]
            else:
                lt, lf, le = self.lab(), self.lab(), self.lab()
                c = self.expr(s[1])
                self.ir += [("br", c, lt, lf), ("label", lt)]
                self.stmts(s[2]); self.ir += [("jmp", le), ("label", lf)]
                self.stmts(s[3]); self.ir += [("label", le)]


def compile_to_ir(src):
    g = Gen(); g.stmts(parse(src)); return g.ir


# --------------------------------------------------------------------- semantics
def tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def ev(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return tdiv(a, b)
    if op == "%": return a - tdiv(a, b) * b
    if op == "<<": return a << b
    return int({"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b, "==": a == b, "!=": a != b}[op])


def run_vm(ir, max_steps=50_000_000):
    """Execute IR. Returns (printed values, executed instruction count)."""
    labels = {ins[1]: i for i, ins in enumerate(ir) if ins[0] == "label"}
    env, out, pc, steps = {}, [], 0, 0
    val = lambda x: x if isinstance(x, int) else env[x]
    while pc < len(ir):
        ins = ir[pc]; op = ins[0]; pc += 1
        if op == "label":
            continue
        steps += 1
        if steps > max_steps:
            raise RuntimeError("step limit")
        if op == "copy": env[ins[1]] = val(ins[2])
        elif op == "bin": env[ins[1]] = ev(ins[2], val(ins[3]), val(ins[4]))
        elif op == "print": out.append(val(ins[1]))
        elif op == "jmp": pc = labels[ins[1]]
        elif op == "br": pc = labels[ins[2] if val(ins[1]) != 0 else ins[3]]
    return out, steps


# ---------------------------------------------------------------------- C backend
def variables(ir):
    vs = []
    for ins in ir:
        for x in ((ins[1],) if ins[0] in ("copy", "bin") else ()) + tuple(u for u in uses(ins)):
            if isinstance(x, str) and x not in vs:
                vs.append(x)
    return vs


def uses(ins):
    op = ins[0]
    if op == "copy": return [ins[2]] if isinstance(ins[2], str) else []
    if op == "bin": return [x for x in (ins[3], ins[4]) if isinstance(x, str)]
    if op == "print": return [ins[1]] if isinstance(ins[1], str) else []
    if op == "br": return [ins[1]] if isinstance(ins[1], str) else []
    return []


def to_c(ir):
    lines = ["#include <stdio.h>", "int main(void){"]
    for v in variables(ir):
        lines.append(f"long long {v}=0;")
    for ins in ir:
        op = ins[0]
        if op == "label": lines.append(f"{ins[1]}:;")
        elif op == "jmp": lines.append(f"goto {ins[1]};")
        elif op == "br": lines.append(f"if ({ins[1]}) goto {ins[2]}; else goto {ins[3]};")
        elif op == "copy": lines.append(f"{ins[1]}={ins[2]};")
        elif op == "print": lines.append(f'printf("%lld\\n",(long long)({ins[1]}));')
        else:
            lines.append(f"{ins[1]}={ins[3]} {ins[2]} {ins[4]};" if ins[2] != "<<" else f"{ins[1]}=(long long)((unsigned long long){ins[3]} << {ins[4]});")
    lines += ["return 0;", "}"]
    return "\n".join(lines)
