"""Exact TSP via Held-Karp dynamic programming (O(n^2 * 2^n))."""


def held_karp(dist):
    """Exact travelling-salesman tour by Held-Karp dynamic programming.
    dist: n x n list of lists (or numpy array) of non-negative numbers, n >= 1 (may be asymmetric).
    Return (cost, tour): the minimum total cost of a cycle that starts at city 0, visits every city exactly once
    and returns to 0, and the tour as a list of city indices starting with 0 (length n, without repeating 0 at the end).
    For n == 1 return (0, [0]).
    """
    n = len(dist)
    if n == 0:
        return (0, [])
    if n == 1:
        return (0, [0])

    # City 0 is the fixed start. Cities 1..n-1 are represented by bits 0..m-1.
    m = n - 1
    full = 1 << m
    INF = float("inf")

    # dp[mask][j]: minimum cost of a path starting at city 0, visiting exactly the
    # cities in mask (bit k -> city k+1), and ending at city j+1 (bit j must be set in mask).
    dp = [[INF] * m for _ in range(full)]
    parent = [[-1] * m for _ in range(full)]

    for j in range(m):
        dp[1 << j][j] = dist[0][j + 1]

    for mask in range(1, full):
        row = dp[mask]
        for j in range(m):
            if not (mask >> j) & 1:
                continue
            cur = row[j]
            if cur == INF:
                continue
            dist_from_j = dist[j + 1]
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nmask = mask | (1 << k)
                cand = cur + dist_from_j[k + 1]
                if cand < dp[nmask][k]:
                    dp[nmask][k] = cand
                    parent[nmask][k] = j

    last = full - 1
    best = INF
    best_j = -1
    for j in range(m):
        c = dp[last][j] + dist[j + 1][0]
        if c < best:
            best = c
            best_j = j

    # Reconstruct the tour by walking the parent pointers backwards.
    rev = []
    mask = last
    j = best_j
    while j != -1:
        rev.append(j + 1)
        pj = parent[mask][j]
        mask ^= 1 << j
        j = pj

    tour = [0] + rev[::-1]
    return best, tour
