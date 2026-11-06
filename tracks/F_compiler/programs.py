"""Benchmark programs in the toy language (N is substituted). Values are kept small so that
long long arithmetic in C never overflows and matches the VM exactly."""

PROGRAMS = {
    # constants that can be folded and propagated
    "const_heavy": """
let a = 3 * 4; let b = a + 5; let c = b * 2 - a; let i = 0; let s = 0;
while (i < N) { s = (s + a * b + c + (2 * 8)) % 1000003; i = i + 1; }
print s;""",
    # the same expression computed repeatedly
    "cse_heavy": """
let x = 7; let y = 9; let i = 0; let s = 0;
while (i < N) {
  let u = (x * y + i) * (x * y + i);
  let v = (x * y + i) + (x * y + i) * 3;
  s = (s + u % 97 + v % 89) % 1000003; i = i + 1; }
print s;""",
    # multiplications by powers of two
    "strength_heavy": """
let i = 0; let s = 0;
while (i < N) { s = (s + i * 8 + i * 4 + i * 16 + (i + 1) * 2) % 1000003; i = i + 1; }
print s;""",
    # work whose result is never used
    "dead_code": """
let i = 0; let s = 0;
while (i < N) {
  let waste1 = i * i * i; let waste2 = waste1 + 99; let waste3 = waste2 * 3;
  s = (s + i) % 1000003; i = i + 1; }
print s;""",
    # a mix, including a branch the compiler can resolve
    "mixed": """
let a = 5; let b = 3; let debug = 0; let i = 0; let s = 0;
while (i < N) {
  let k = (a + b) * (a + b);
  if (debug) { print k; } else { s = (s + k * 8 + i * 2) % 1000003; }
  let junk = k * 31;
  i = i + 1; }
print s;""",
    # control: little for the optimizer to do
    "control_primes": """
let n = 2; let count = 0;
while (n < N) {
  let d = 2; let prime = 1;
  while (d * d <= n) { if (n % d == 0) { prime = 0; d = n; } d = d + 1; }
  count = count + prime; n = n + 1; }
print count;""",
}

SIZES = {"const_heavy": 30_000_000, "cse_heavy": 20_000_000, "strength_heavy": 30_000_000,
         "dead_code": 30_000_000, "mixed": 30_000_000, "control_primes": 1_200_000}
SMALL = {k: (n if k == "control_primes" else 2000) for k, n in SIZES.items()}
SMALL["control_primes"] = 300


def source(name, n):
    return PROGRAMS[name].replace("N", str(n))
