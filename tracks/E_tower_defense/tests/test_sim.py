"""Hand-computed checks for the simulator (see comments for the arithmetic)."""
import random

import pytest

from td.baselines import random_placement
from td.maps import make_wave, snake_map, straight_map
from td.sim import DEFAULT_TYPES, Enemy, TowerType, simulate, validate_placement
from td.surrogate import Surrogate

M = straight_map(10, 3)          # path (0,0)..(9,0)
T = {
    "dmg": TowerType("dmg", cost=10, range=1.5, dps=10.0),
    "frost": TowerType("frost", cost=5, range=1.5, dps=0.0, slow=0.5),
}


def test_no_towers_everything_leaks():
    r = simulate(M, {}, T, [Enemy(hp=100, speed=1.0, spawn=0)])
    assert (r.damage, r.leaked, r.leaked_hp, r.killed) == (0, 1, 100, 0)


def test_in_range_for_three_ticks():
    # tower (4,1), range 1.5 covers path cells 3,4,5; speed 1 -> fires at ticks 3,4,5
    r = simulate(M, {(4, 1): "dmg"}, T, [Enemy(100, 1.0, 0)])
    assert r.damage == 30 and r.leaked == 1 and r.leaked_hp == 70


def test_kill_stops_leak():
    # hp 20: 10 dmg at tick 3 and tick 4 -> dead
    r = simulate(M, {(4, 1): "dmg"}, T, [Enemy(20, 1.0, 0)])
    assert (r.damage, r.killed, r.leaked) == (20, 1, 0)


def test_overkill_not_counted():
    r = simulate(M, {(4, 1): "dmg"}, T, [Enemy(15, 1.0, 0)])
    assert r.damage == 15 and r.killed == 1


def test_slow_tower_adds_exposure():
    # frost (3,1) slows on cells 2,3,4. Enemy positions: 0,1,2,2.5,3,3.5,4,4.5,5
    # damage tower fires at ticks with cell in {3,4,5}: t=4..8 -> 5 shots = 50
    p = {(4, 1): "dmg", (3, 1): "frost"}
    r = simulate(M, p, T, [Enemy(1000, 1.0, 0)])
    assert r.damage == 50


def test_targets_furthest_enemy():
    # two enemies in range together; one tower must hit the one further along
    w = [Enemy(1000, 1.0, 0), Enemy(1000, 1.0, 0)]
    r = simulate(M, {(4, 1): "dmg"}, T, w)
    assert r.damage == 30      # same spawn tick: tie -> same total shots, no crash


def test_determinism():
    wave = make_wave(3, 20)
    g = snake_map(11, 7)
    p = random_placement(g, DEFAULT_TYPES, 100, random.Random(1))
    a = simulate(g, p, DEFAULT_TYPES, wave)
    b = simulate(g, p, DEFAULT_TYPES, wave)
    assert a == b


def test_validate_rejects_bad_placements():
    with pytest.raises(ValueError):
        validate_placement(M, {(3, 0): "dmg"}, T, 100)       # on the path
    with pytest.raises(ValueError):
        validate_placement(M, {(3, 1): "dmg", (4, 1): "dmg"}, T, 15)   # over budget
    with pytest.raises(ValueError):
        validate_placement(M, {(3, 1): "laser"}, T, 100)     # unknown type


def test_bad_enemy_rejected():
    with pytest.raises(ValueError):
        simulate(M, {}, T, [Enemy(10, 0.0, 0)])


def test_random_baseline_respects_constraints():
    g = snake_map(15, 11)
    for s in range(20):
        p = random_placement(g, DEFAULT_TYPES, 120, random.Random(s))
        validate_placement(g, p, DEFAULT_TYPES, 120)


def test_surrogate_hand_value():
    # tower covers 3 path cells, dps 10, speed 1 -> 30 damage, hp 100 -> 30
    s = Surrogate(M, T, [Enemy(100, 1.0, 0)])
    assert s.value({(4, 1): "dmg"}) == 30
    # capped by hp
    assert Surrogate(M, T, [Enemy(20, 1.0, 0)]).value({(4, 1): "dmg"}) == 20


def test_surrogate_tower_capacity_matches_simulator():
    # two enemies in range together: each could take 30, but the tower fires only
    # 3 ticks -> 30 total. The simulator gives the same number (test_targets_furthest_enemy).
    w = [Enemy(1000, 1.0, 0), Enemy(1000, 1.0, 0)]
    assert Surrogate(M, T, w).value({(4, 1): "dmg"}) == pytest.approx(30)
    assert simulate(M, {(4, 1): "dmg"}, T, w).damage == 30


def test_surrogate_is_monotone_submodular_on_random_sets():
    g = snake_map(11, 7)
    wave = make_wave(0, 12)
    s = Surrogate(g, DEFAULT_TYPES, wave)
    rng = random.Random(5)
    cells = g.buildable()
    names = list(DEFAULT_TYPES)
    for _ in range(60):
        B = {c: rng.choice(names) for c in rng.sample(cells, 8)}
        A = dict(list(B.items())[:4])
        x = rng.choice([c for c in cells if c not in B])
        n = rng.choice(names)
        gainA = s.value({**A, x: n}) - s.value(A)
        gainB = s.value({**B, x: n}) - s.value(B)
        assert gainA >= gainB - 1e-9          # diminishing returns
        assert s.value(B) >= s.value(A) - 1e-9   # monotone
