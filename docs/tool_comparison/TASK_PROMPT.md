# Task given to every AI tool (paste this exactly, in a fresh chat)

Write a Python 3 module with one function:

```python
def held_karp(dist):
    """Exact travelling-salesman tour by Held-Karp dynamic programming.
    dist: n x n list of lists (or numpy array) of non-negative numbers, n >= 1 (may be asymmetric).
    Return (cost, tour): the minimum total cost of a cycle that starts at city 0, visits every city exactly once
    and returns to 0, and the tour as a list of city indices starting with 0 (length n, without repeating 0 at the end).
    For n == 1 return (0, [0])."""
```

Requirements: O(n^2 * 2^n) time, standard library and numpy only, no external solvers. It must work for n up to 15 in a few seconds. Give only the code.
