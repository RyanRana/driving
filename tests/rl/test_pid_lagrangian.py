"""Unit tests for the pure PID-Lagrangian controller (experiment).

These are pure-math tests — no JAX, no env, no network. They pin the three
behaviours the controller must guarantee for safe-RL dual ascent:
  * constant positive error → λ rises monotonically then saturates at λ_max;
  * error crossing to negative → derivative damps and the integral unwinds so
    λ decays back toward 0;
  * λ is never negative regardless of input.
"""
from __future__ import annotations

from smoothride.rl.pid_lagrangian import (
    CHANNEL_SPEC,
    GRADED_GAINS,
    HARD_GAINS,
    PIDGains,
    PIDState,
    pid_step,
)


def test_constant_positive_error_lambda_rises_then_saturates() -> None:
    """A persistent above-target cost must push λ up and saturate at λ_max."""
    gains = PIDGains(kp=1.0, ki=5.0, kd=2.0)
    state = PIDState()
    lam_max = 400.0
    lams = []
    # error=0.1, Ki=5 → integral cap 80 reached after ~800 steps; run long enough.
    for _ in range(2000):
        lam, state = pid_step(state, mean_cost=0.1, target=0.0,
                              gains=gains, lam_max=lam_max)
        lams.append(lam)

    # Monotone non-decreasing while climbing (integral keeps accumulating).
    for a, b in zip(lams[:50], lams[1:51]):
        assert b >= a - 1e-9, "λ must not decrease under constant positive error"
    # Eventually saturates at λ_max and stays there.
    assert lams[-1] == lam_max
    assert max(lams) <= lam_max + 1e-9


def test_error_crossing_zero_damps_and_decays() -> None:
    """Once cost drops below target, λ must decay back toward zero."""
    gains = PIDGains(kp=1.0, ki=5.0, kd=2.0)
    state = PIDState()
    # Build up λ with several above-target steps.
    for _ in range(20):
        lam_high, state = pid_step(state, mean_cost=0.2, target=0.05, gains=gains)
    assert lam_high > 0.0

    # Now cost is well below target → integral unwinds, λ falls.
    lam_prev = lam_high
    for _ in range(200):
        lam, state = pid_step(state, mean_cost=0.0, target=0.05, gains=gains)
        assert lam <= lam_prev + 1e-9, "λ must not rise once cost is below target"
        lam_prev = lam
    assert lam_prev < lam_high, "λ must have decayed below its peak"


def test_rising_only_derivative_does_not_fire_on_falling_error() -> None:
    """The derivative term is rising-only: a falling error contributes 0 from D."""
    # Pure-D channel: isolate the derivative term.
    gains = PIDGains(kp=0.0, ki=0.0, kd=10.0)
    state = PIDState(lam=0.0, integral=0.0, prev_error=1.0)
    # error goes from 1.0 -> 0.2 (falling). deriv = max(0, 0.2-1.0) = 0.
    lam, _ = pid_step(state, mean_cost=0.2, target=0.0, gains=gains)
    assert lam == 0.0

    # error rises 0.0 -> 0.5 → deriv fires.
    state2 = PIDState(lam=0.0, integral=0.0, prev_error=0.0)
    lam2, _ = pid_step(state2, mean_cost=0.5, target=0.0, gains=gains)
    assert lam2 > 0.0


def test_lambda_never_negative() -> None:
    """Deeply-below-target cost must clip λ at 0, never go negative."""
    gains = PIDGains(kp=1.0, ki=5.0, kd=2.0)
    state = PIDState(lam=10.0, integral=2.0, prev_error=0.0)
    lam, state = pid_step(state, mean_cost=0.0, target=5.0, gains=gains)
    assert lam >= 0.0
    # Hammer it with hugely-negative errors; λ stays pinned at 0.
    for _ in range(50):
        lam, state = pid_step(state, mean_cost=0.0, target=100.0, gains=gains)
        assert lam == 0.0


def test_anti_windup_caps_integral_contribution() -> None:
    """Integral is clamped so Ki*integral can't exceed λ_max (anti-windup)."""
    gains = PIDGains(kp=0.0, ki=5.0, kd=0.0)
    state = PIDState()
    for _ in range(1000):
        _, state = pid_step(state, mean_cost=1.0, target=0.0,
                            gains=gains, lam_max=400.0)
    assert gains.ki * state.integral <= 400.0 + 1e-6


def test_returns_new_state_without_mutation() -> None:
    """pid_step must return a NEW state and not mutate the input (immutability)."""
    gains = PIDGains(kp=1.0, ki=5.0, kd=2.0)
    state = PIDState()
    lam, new_state = pid_step(state, mean_cost=0.1, target=0.0, gains=gains)
    assert new_state is not state
    assert state.lam == 0.0 and state.integral == 0.0 and state.prev_error == 0.0
    assert new_state.lam == lam


def test_channel_spec_covers_five_constraints() -> None:
    """CHANNEL_SPEC must define the five experiment channels with valid targets."""
    assert set(CHANNEL_SPEC) == {
        "car_crash", "ped_hit", "car_risk", "ped_yield", "lane",
    }
    # Binary collisions track the zero target; graded hinges track soft.
    assert CHANNEL_SPEC["car_crash"][1] == "zero"
    assert CHANNEL_SPEC["ped_hit"][1] == "zero"
    for ch in ("car_risk", "ped_yield", "lane"):
        assert CHANNEL_SPEC[ch][1] == "soft"
    # Ki mirrors the champion step sizes.
    assert CHANNEL_SPEC["car_crash"][0] == HARD_GAINS
    assert HARD_GAINS.ki == 5.0
    assert GRADED_GAINS.ki == 2.0
