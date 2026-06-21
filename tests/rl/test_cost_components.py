"""Tests for the graded car-collision-risk hinge and hard/soft cost components.

v2 Task 2 (TDD): these tests are written BEFORE the implementation.

The crash cost is a SPARSE BINARY terminal spike — bad for credit assignment. We
add a GRADED car-collision-risk hinge (`car_risk_cost`, mirroring `ped_yield_cost`)
and expose collision components separately so a later task can split HARD
(collisions→0) from SOFT (graded) constraints.
"""
from __future__ import annotations

import numpy as np
import pytest

from smoothride.rl.verifier import (
    CAR_RISK_RADIUS,
    COLLISION_RADIUS,
    CRUISE_CAP,
    car_risk_cost,
    hard_cost,
    soft_cost,
    step_cost,
    step_cost_components,
)


# ---------------------------------------------------------------------------
# car_risk_cost — the graded hinge over OTHER cars
# ---------------------------------------------------------------------------
def _risk(
    cars: list[list[float]],
    speeds: list[float],
    grace: list[int] | None = None,
    r_risk: float = CAR_RISK_RADIUS,
    collision_radius: float = COLLISION_RADIUS,
    cruise_cap: float = CRUISE_CAP,
) -> np.ndarray:
    """Helper: a single timestep with N cars → (N,) cost vector for that step."""
    n = len(cars)
    pos = np.array([cars], np.float32)            # (1, N, 2)
    spd = np.array([speeds], np.float32)          # (1, N)
    grace_arr = np.array([grace if grace is not None else [0] * n], np.int32)  # (1, N)
    return car_risk_cost(pos, spd, grace_arr, r_risk, collision_radius, cruise_cap)[0]


def test_zero_when_single_car() -> None:
    assert np.all(_risk([[0.0, 0.0]], [7.0]) == 0.0)


def test_zero_when_all_far() -> None:
    out = _risk([[0.0, 0.0], [100.0, 0.0]], [7.0, 7.0])
    assert np.all(out == 0.0)


def test_zero_when_stopped() -> None:
    # cars adjacent but both stopped → no closing risk
    out = _risk([[0.0, 0.0], [3.0, 0.0]], [0.0, 0.0])
    assert np.all(out == 0.0)


def test_ramps_with_proximity() -> None:
    near = _risk([[0.0, 0.0], [3.0, 0.0]], [7.0, 7.0])[0]
    far = _risk([[0.0, 0.0], [6.0, 0.0]], [7.0, 7.0])[0]
    assert near > far > 0.0


def test_ramps_with_speed() -> None:
    fast = _risk([[0.0, 0.0], [4.0, 0.0]], [7.0, 0.0])[0]
    slow = _risk([[0.0, 0.0], [4.0, 0.0]], [2.0, 0.0])[0]
    assert fast > slow > 0.0


def test_graded_midpoint_strictly_between_0_and_1() -> None:
    # midpoint of [collision_radius=2.2, r_risk=7.0] = 4.6 m, full speed
    mid = _risk([[0.0, 0.0], [4.6, 0.0]], [7.0, 0.0])[0]
    assert 0.0 < mid < 1.0


def test_max_over_multiple_cars() -> None:
    # one near (3 m) + one far (6 m); max → equals the near-only value
    multi = _risk([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], [7.0, 0.0, 0.0])[0]
    near_only = _risk([[0.0, 0.0], [3.0, 0.0]], [7.0, 0.0])[0]
    assert abs(multi - near_only) < 1e-6


def test_excludes_self() -> None:
    # a single car cannot be its own collision partner (diagonal excluded)
    assert _risk([[0.0, 0.0]], [7.0])[0] == 0.0


def test_excludes_spawn_immune_partners() -> None:
    # neighbour at 3 m is spawn-immune → contributes no risk to the ego
    out = _risk([[0.0, 0.0], [3.0, 0.0]], [7.0, 7.0], grace=[0, 5])
    assert out[0] == 0.0


def test_asserts_when_radii_equal() -> None:
    with pytest.raises(AssertionError):
        _risk([[0.0, 0.0], [3.0, 0.0]], [7.0, 7.0],
              r_risk=2.2, collision_radius=2.2)


def test_handles_n_less_than_two() -> None:
    # N=0 and N=1 both return all-zeros of the right shape
    pos0 = np.zeros((2, 0, 2), np.float32)
    spd0 = np.zeros((2, 0), np.float32)
    grace0 = np.zeros((2, 0), np.int32)
    out0 = car_risk_cost(pos0, spd0, grace0)
    assert out0.shape == (2, 0)

    pos1 = np.zeros((2, 1, 2), np.float32)
    spd1 = np.full((2, 1), 7.0, np.float32)
    grace1 = np.zeros((2, 1), np.int32)
    out1 = car_risk_cost(pos1, spd1, grace1)
    assert out1.shape == (2, 1)
    assert np.all(out1 == 0.0)


# ---------------------------------------------------------------------------
# step_cost_components — the 7-term decomposition
# ---------------------------------------------------------------------------
def _components_inputs(T: int = 2, N: int = 2):
    pos = np.zeros((T, N, 2), np.float32)
    pos[..., 1] = -3.5 * 0.5            # on lane-0 centerline
    seg_start = np.zeros((T, N, 2), np.float32)
    seg_end = np.zeros((T, N, 2), np.float32)
    seg_end[..., 0] = 1.0
    lane_count = np.ones((T, N), np.float32)
    heading = np.zeros((T, N), np.float32)
    speed = np.ones((T, N), np.float32)
    spawn_grace = np.zeros((T, N), np.int32)
    car_crashed = np.zeros((T, N), np.float32)
    ped_hit = np.zeros((T, N), np.float32)
    return dict(
        pos=pos, seg_start=seg_start, seg_end=seg_end, lane_count=lane_count,
        lane_width=3.5, heading=heading, speed=speed, spawn_grace=spawn_grace,
        car_crashed=car_crashed, ped_hit=ped_hit,
    )


def test_components_returns_all_seven_keys() -> None:
    comp = step_cost_components(**_components_inputs())
    assert set(comp) == {
        "car_crash", "ped_hit", "off_lane", "wrong_way",
        "over_cap", "ped_yield", "car_risk",
    }
    for k, v in comp.items():
        assert v.shape == (2, 2), f"{k} has wrong shape {v.shape}"


def test_components_lane_terms_match_legacy_step_cost() -> None:
    """off_lane + wrong_way + over_cap + ped_yield from components must equal the
    corresponding portion of the legacy step_cost (predicates unchanged)."""
    T, N = 2, 2
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 5, (T, N, 2)).astype(np.float32)
    seg_start = np.zeros((T, N, 2), np.float32)
    seg_end = np.zeros((T, N, 2), np.float32)
    seg_end[..., 0] = 10.0
    lane_count = np.full((T, N), 2.0, np.float32)
    heading = rng.uniform(-np.pi, np.pi, (T, N)).astype(np.float32)
    speed = rng.uniform(0, 12, (T, N)).astype(np.float32)
    spawn_grace = np.zeros((T, N), np.int32)
    crashed = np.zeros((T, N), np.float32)
    ped_hit = np.zeros((T, N), np.float32)
    speed_limit = np.full((T, N), 7.0, np.float32)
    ped_pos = rng.normal(0, 5, (T, 3, 2)).astype(np.float32)
    ped_crossing = np.array([[True, False, True], [False, True, False]])

    comp = step_cost_components(
        pos, seg_start, seg_end, lane_count, 3.5, heading, speed, spawn_grace,
        crashed, ped_hit, speed_limit, ped_pos=ped_pos, ped_crossing=ped_crossing)

    # Legacy step_cost = crash + off_lane + wrong_way + over_cap + ped_yield.
    # With crashed=0, that equals the non-collision component sum (no car_risk).
    legacy = step_cost(
        pos, seg_start, seg_end, lane_count, 3.5, heading, speed, spawn_grace,
        crashed, speed_limit, ped_pos=ped_pos, ped_crossing=ped_crossing)

    portion = (comp["off_lane"] + comp["wrong_way"]
               + comp["over_cap"] + comp["ped_yield"])
    np.testing.assert_allclose(portion, legacy, atol=1e-6)


def test_components_over_cap_zero_without_limit() -> None:
    comp = step_cost_components(**_components_inputs())  # no speed_limit
    assert np.all(comp["over_cap"] == 0.0)


def test_components_ped_yield_zero_without_peds() -> None:
    comp = step_cost_components(**_components_inputs())  # no ped_pos
    assert np.all(comp["ped_yield"] == 0.0)


# ---------------------------------------------------------------------------
# hard_cost / soft_cost partition
# ---------------------------------------------------------------------------
def _fake_components(T: int = 1, N: int = 1, **vals) -> dict[str, np.ndarray]:
    base = {k: np.zeros((T, N), np.float32) for k in (
        "car_crash", "ped_hit", "off_lane", "wrong_way",
        "over_cap", "ped_yield", "car_risk")}
    for k, v in vals.items():
        base[k] = np.full((T, N), v, np.float32)
    return base


def test_hard_cost_weights_ped_higher() -> None:
    car = hard_cost(_fake_components(car_crash=1.0))[0, 0]
    ped = hard_cost(_fake_components(ped_hit=1.0))[0, 0]
    assert car == pytest.approx(1.0)
    assert ped == pytest.approx(3.0)
    assert ped == pytest.approx(3.0 * car)


def test_hard_cost_excludes_graded_terms() -> None:
    # car_risk / ped_yield / lane terms must NOT enter hard_cost
    out = hard_cost(_fake_components(
        car_risk=5.0, ped_yield=5.0, off_lane=5.0, wrong_way=5.0, over_cap=5.0))
    assert np.all(out == 0.0)


def test_soft_cost_includes_graded_excludes_binary() -> None:
    comp = _fake_components(
        car_crash=1.0, ped_hit=1.0,            # binary — excluded
        off_lane=1.0, wrong_way=1.0, over_cap=1.0, ped_yield=1.0, car_risk=1.0)
    out = soft_cost(comp)[0, 0]
    assert out == pytest.approx(5.0)           # the five graded/lane terms


def test_soft_cost_zero_when_only_collisions() -> None:
    out = soft_cost(_fake_components(car_crash=1.0, ped_hit=1.0))
    assert np.all(out == 0.0)
