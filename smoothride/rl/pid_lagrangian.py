"""PID-Lagrangian dual controller (Stooke et al. 2020).

The proven champion (_v4loo) drives each Lagrange multiplier with PURE INTEGRAL
dual ascent::

    lam <- clip(lam + Ki * (mean_cost - target), 0, lam_max)

That I-only controller oscillates: ``lam`` overshoots, the policy over-corrects,
cost dips below target, ``lam`` decays, and violations creep back. Stooke et al.,
*"Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"* (ICML
2020), add proportional (P) and derivative (D) terms so the multiplier responds
to the *current* error magnitude and its *rate of change*, damping the overshoot
while keeping the same integral that made the champion converge.

This module is intentionally pure (no JAX, no env): the controller state is a
small frozen dataclass and ``pid_step`` is a referentially-transparent function
returning a NEW state. That keeps the dual loop trivially testable in isolation
and free of hidden mutation.

Design choices (documented for the experiment):
  * **Integral gain Ki is kept equal to the champion step sizes** (Ki=5.0 hard,
    Ki=2.0 graded) so the experiment stays close to the proven optimum and any
    delta is attributable to the new P/D damping rather than a changed integrator.
  * **Rising-only derivative** ``deriv = max(0, error - prev_error)``: D acts only
    when the constraint is getting WORSE (cost rising toward/over target). When
    cost is improving we let the integral decay ``lam`` naturally instead of
    yanking it down — this is the conservative, stability-favouring choice in
    safe RL (a falling-error D term would aggressively *drop* the price right when
    the policy is finally complying, re-inviting violations). Stooke et al.
    discuss both; rising-only is the safer default for hard safety channels.
  * **Anti-windup**: the integral is clamped to the band that keeps
    ``Ki * integral`` within ``[0, lam_max]``, so a long stretch above target
    can't wind the integral up into a value that takes forever to unwind.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PIDGains:
    """Per-channel PID gains.

    Attributes:
        kp: Proportional gain — responds to the instantaneous error.
        ki: Integral gain — the running sum of error; kept equal to the
            champion's pure-integral step size for continuity.
        kd: Derivative gain — responds to a *rising* error (damping term).
    """

    kp: float
    ki: float
    kd: float


@dataclass(frozen=True)
class PIDState:
    """Immutable controller state carried across iterations.

    Attributes:
        lam: The current Lagrange multiplier (the controller output).
        integral: Running (anti-windup-clamped) sum of error.
        prev_error: Previous step's error, for the derivative term.
    """

    lam: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0


def pid_step(
    state: PIDState,
    mean_cost: float,
    target: float,
    gains: PIDGains,
    lam_max: float = 400.0,
) -> tuple[float, PIDState]:
    """Advance the PID-Lagrangian controller by one dual step.

    Implements (Stooke et al. 2020)::

        error    = mean_cost - target
        integral = clamp(integral + error, integral_lo, integral_hi)   # anti-windup
        deriv    = max(0, error - prev_error)                          # rising-only
        lam      = clip(Kp*error + Ki*integral + Kd*deriv, 0, lam_max)

    The integral band ``[integral_lo, integral_hi]`` is chosen so the integral
    contribution ``Ki*integral`` alone stays within ``[0, lam_max]`` (anti-windup).
    With ``Ki <= 0`` the integral is not clamped (degenerate / unused channel).

    Args:
        state: Current :class:`PIDState` (not mutated).
        mean_cost: Mean per-step cost of this channel over the last rollout.
        target: Constraint target for this channel (e.g. 0.0 for hard collisions).
        gains: Per-channel :class:`PIDGains`.
        lam_max: Upper clip on the multiplier (lower clip is always 0).

    Returns:
        ``(new_lam, new_state)`` — the multiplier to price this channel's cost
        with, and the NEW controller state to carry to the next iteration.
        ``new_state.lam == new_lam``.
    """
    error = mean_cost - target

    integral = state.integral + error
    if gains.ki > 0.0:
        integral = min(lam_max / gains.ki, max(0.0, integral))

    deriv = max(0.0, error - state.prev_error)

    raw = gains.kp * error + gains.ki * integral + gains.kd * deriv
    lam = min(lam_max, max(0.0, raw))

    new_state = replace(state, lam=lam, integral=integral, prev_error=error)
    return lam, new_state


# ---------------------------------------------------------------------------
# Default per-channel gains for the experiment.
#
# Five channels (see ppo.verifier_costs_split): two HARD (binary collisions,
# target 0) and three GRADED (car-risk / ped-yield / lane hinges, target
# soft_target). Ki mirrors the champion step sizes (5.0 hard, 2.0 graded). Kp~1
# adds proportional responsiveness; Kd damps a rising error. Hard channels get a
# slightly stronger Kd (2.0) — overshoot on a safety-critical collision price is
# the most costly failure mode, so we damp it harder.
# ---------------------------------------------------------------------------
HARD_GAINS = PIDGains(kp=1.0, ki=5.0, kd=2.0)
GRADED_GAINS = PIDGains(kp=1.0, ki=2.0, kd=1.0)

#: Channel -> (gains, target_key). ``target_key`` is resolved at runtime:
#: ``"zero"`` -> crash_target (0.0); ``"soft"`` -> soft_target.
CHANNEL_SPEC: dict[str, tuple[PIDGains, str]] = {
    "car_crash": (HARD_GAINS, "zero"),
    "ped_hit": (HARD_GAINS, "zero"),
    "car_risk": (GRADED_GAINS, "soft"),
    "ped_yield": (GRADED_GAINS, "soft"),
    "lane": (GRADED_GAINS, "soft"),
}
