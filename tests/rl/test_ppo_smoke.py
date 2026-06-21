"""End-to-end PPO training smoke test (v2 Task 3).

Exercises the full collect → verifier_cost → update loop with the dual-channel
(hard/soft) Lagrangian and the Deep Sets dict-obs policy.

The tests assert:
  * collect produces ``car_crash`` and ``ped_hit`` arrays in the batch.
  * verifier_cost returns a ``(cost_hard, cost_soft)`` tuple each shaped (B, T, N).
  * update runs end-to-end with dual lam_hard / lam_soft and emits finite metrics.
  * car_ped_rate and car_car_rate are computable from the batch.
"""
import numpy as np
import jax
import jax.numpy as jnp

from smoothride.rl import ppo
from tests.env.test_kinematic_peds import _env


def test_collect_has_collision_subcomponents() -> None:
    """collect must expose car_crash and ped_hit in the batch."""
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(1), cfg.n_worlds)

    assert "car_crash" in batch, "batch must contain car_crash"
    assert "ped_hit" in batch, "batch must contain ped_hit"

    B, T, N = np.asarray(batch["reward"]).shape
    assert np.asarray(batch["car_crash"]).shape == (B, T, N), (
        f"car_crash shape mismatch: {np.asarray(batch['car_crash']).shape}"
    )
    assert np.asarray(batch["ped_hit"]).shape == (B, T, N), (
        f"ped_hit shape mismatch: {np.asarray(batch['ped_hit']).shape}"
    )


def test_verifier_cost_dual_channel_shape() -> None:
    """verifier_cost must return (cost_hard, cost_soft) each of shape (B, T, N)."""
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(2), cfg.n_worlds)

    result = ppo.verifier_cost(env, batch, w_carped=3.0)
    assert isinstance(result, tuple) and len(result) == 2, (
        "verifier_cost must return a 2-tuple (cost_hard, cost_soft)"
    )
    cost_hard, cost_soft = result

    B, T, N = np.asarray(batch["reward"]).shape
    assert cost_hard.shape == (B, T, N), (
        f"cost_hard shape {cost_hard.shape} != ({B}, {T}, {N})"
    )
    assert cost_soft.shape == (B, T, N), (
        f"cost_soft shape {cost_soft.shape} != ({B}, {T}, {N})"
    )
    assert np.all(np.isfinite(cost_hard)), "cost_hard contains non-finite values"
    assert np.all(np.isfinite(cost_soft)), "cost_soft contains non-finite values"
    assert np.all(cost_hard >= 0.0), "cost_hard must be non-negative"
    assert np.all(cost_soft >= 0.0), "cost_soft must be non-negative"


def test_one_ppo_iteration_runs_end_to_end() -> None:
    """One full PPO iteration must complete and emit finite metrics.

    Config: n_worlds=2 keeps the JIT-compiled scan small.
    epochs=1, minibatches=2 are enough to exercise the minibatch
    loop without a long compile.  Total flat obs size =
    n_worlds * max_steps * n_agents = 2 * 300 * 4 = 2 400, which
    is divisible by minibatches=2 (mb_size=1200).
    """
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)

    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(1), cfg.n_worlds)
    cost_hard, cost_soft = ppo.verifier_cost(env, batch, w_carped=3.0)
    batch = {**batch, "cost_hard": cost_hard, "cost_soft": cost_soft}

    ts2, metrics = ppo.update(env, cfg, ts, batch, lam_hard=1.0, lam_soft=0.5)

    assert jnp.isfinite(metrics["loss"]), (
        f"loss is not finite: {metrics['loss']}"
    )
    assert jnp.isfinite(metrics["ep_reward"]), (
        f"ep_reward is not finite: {metrics['ep_reward']}"
    )
    # Sanity: update must return a valid TrainState (not the same object).
    assert ts2 is not ts, "update must return a new TrainState"


def test_car_ped_rate_and_car_car_rate_accessible() -> None:
    """car_ped_rate and car_car_rate must be computable from the batch."""
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(3), cfg.n_worlds)

    car_ped_rate = float(np.asarray(batch["ped_hit"]).mean())
    car_car_rate = float(np.asarray(batch["car_crash"]).mean())

    assert np.isfinite(car_ped_rate), f"car_ped_rate not finite: {car_ped_rate}"
    assert np.isfinite(car_car_rate), f"car_car_rate not finite: {car_car_rate}"
    assert 0.0 <= car_ped_rate <= 1.0, f"car_ped_rate out of [0,1]: {car_ped_rate}"
    assert 0.0 <= car_car_rate <= 1.0, f"car_car_rate out of [0,1]: {car_car_rate}"


def test_update_backward_compat_single_lam() -> None:
    """update with legacy lam= (no cost_hard/cost_soft) must still work."""
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(4), cfg.n_worlds)
    # Legacy: keep "cost" (just_crashed) in the batch, pass lam= keyword.
    ts2, metrics = ppo.update(env, cfg, ts, batch, lam=1.0)
    assert jnp.isfinite(metrics["loss"]), (
        f"legacy single-lam path: loss not finite: {metrics['loss']}"
    )
