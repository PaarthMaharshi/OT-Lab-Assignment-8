"""Claude Code (Claude Opus 5.5) answer to TASK_PROMPT.md, first attempt, unedited."""


def held_karp(dist):
    n = len(dist)
    if n == 1:
        return 0, [0]
    full = 1 << (n - 1)                       # subsets of cities 1..n-1; bit i-1 stands for city i
    INF = float("inf")
    dp = [[INF] * n for _ in range(full)]     # dp[S][j]: cheapest path 0 -> ... -> j visiting exactly S (j in S)
    parent = [[-1] * n for _ in range(full)]
    for j in range(1, n):
        dp[1 << (j - 1)][j] = dist[0][j]
    for S in range(1, full):
        for j in range(1, n):
            bj = 1 << (j - 1)
            if not S & bj or dp[S][j] == INF:
                continue
            base = dp[S][j]
            for k in range(1, n):
                bk = 1 << (k - 1)
                if S & bk:
                    continue
                c = base + dist[j][k]
                T = S | bk
                if c < dp[T][k]:
                    dp[T][k] = c
                    parent[T][k] = j
    S = full - 1
    cost, last = min((dp[S][j] + dist[j][0], j) for j in range(1, n))
    tour = []
    while last != -1:
        tour.append(last)
        prev = parent[S][last]
        S ^= 1 << (last - 1)
        last = prev
    return cost, [0] + tour[::-1]
