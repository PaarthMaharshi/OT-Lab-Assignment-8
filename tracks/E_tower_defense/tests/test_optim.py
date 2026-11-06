import itertools
import random

from td.maps import make_wave, snake_map, straight_map
from td.optim import anneal, greedy, ilp
from td.sim import DEFAULT_TYPES, Enemy, TowerType, simulate, validate_placement
from td.surrogate import Surrogate


def _brute_force(sur, types, budget):
    """Exhaustive optimum of the surrogate on a tiny instance."""
    best = 0.0
    opts = [None] + [n for n in sur.tnames if types[n].dps > 0]
    for combo in itertools.product(opts, repeat=len(sur.cells)):
        p = {c: n for c, n in zip(sur.cells, combo) if n}
        if sum(types[n].cost for n in p.values()) <= budget:
            best = max(best, sur.value(p))
    return best


def test_ilp_matches_brute_force_on_tiny_instance():
    g = straight_map(8, 2)                                   # 8 buildable cells
    types = {"a": TowerType("a", 10, 1.5, 10.0), "b": TowerType("b", 20, 2.0, 25.0)}
    wave = [Enemy(40, 1.0, 0), Enemy(90, 0.5, 1)]
    sur = Surrogate(g, types, wave)
    for budget in (10, 30, 40):
        p, optimal = ilp(sur, types, budget)
        assert optimal
        validate_placement(g, p, types, budget)
        assert abs(sur.value(p) - _brute_force(sur, types, budget)) < 1e-6


def test_greedy_valid_and_not_better_than_ilp():
    g = snake_map(11, 7)
    wave = make_wave(0, 15)
    sur = Surrogate(g, DEFAULT_TYPES, wave)
    for budget in (20, 40, 60):
        pg = greedy(sur, DEFAULT_TYPES, budget)
        validate_placement(g, pg, DEFAULT_TYPES, budget)
        pi, opt = ilp(sur, DEFAULT_TYPES, budget)
        assert opt
        assert sur.value(pi) >= sur.value(pg) - 1e-6          # exact >= heuristic
        # knapsack bound for cost-benefit greedy + best single: (1/2)(1-1/e)
        assert sur.value(pg) >= 0.316 * sur.value(pi) - 1e-6


def test_greedy_never_uses_zero_dps_tower_on_surrogate():
    g = snake_map(11, 7)
    sur = Surrogate(g, DEFAULT_TYPES, make_wave(0, 15))
    assert "frost" not in greedy(sur, DEFAULT_TYPES, 60).values()


def test_anneal_deterministic_valid_and_improves():
    g = snake_map(11, 7)
    wave = make_wave(1, 15)
    a, sa = anneal(g, DEFAULT_TYPES, wave, 30, evals=150, seed=4)
    b, sb = anneal(g, DEFAULT_TYPES, wave, 30, evals=150, seed=4)
    assert a == b and sa == sb
    validate_placement(g, a, DEFAULT_TYPES, 30)
    start, _ = anneal(g, DEFAULT_TYPES, wave, 30, evals=1, seed=4)
    assert simulate(g, a, DEFAULT_TYPES, wave).damage >= simulate(g, start, DEFAULT_TYPES, wave).damage
