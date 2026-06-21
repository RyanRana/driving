"""PPO-side tests for the PID-Lagrangian experiment.

Exercises:
  * ppo.verifier_costs_split returns the five per-constraint channels, each
    (B, T, N), non-negative, and partitioning exactly into the legacy
    hard/soft channels (so the experiment stays comparable to the champion);
  * ppo.update with an explicit ``penalty`` array runs end-to-end and emits
    finite metrics.
"""
import jax
import jax.numpy as jnp
import numpy as np

from smoothride.rl import ppo
from tests.env.test_kinematic_peds import _env


def _collect(seed: int):
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)
    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(seed), cfg.n_worlds)
    return env, cfg, ts, batch


def test_verifier_costs_split_shapes_and_keys() -> None:
    env, _, _, batch = _collect(11)
    channels = ppo.verifier_costs_split(env, batch, w_carped=3.0)

    assert set(channels) == {
        "car_crash", "ped_hit", "car_risk", "ped_yield", "lane",
    }
    B, T, N = np.asarray(batch["reward"]).shape
    for k, v in channels.items():
        assert v.shape == (B, T, N), f"{k} shape {v.shape} != {(B, T, N)}"
        assert np.all(np.isfinite(v)), f"{k} has non-finite values"
        assert np.all(v >= 0.0), f"{k} must be non-negative"


def test_split_partitions_into_hard_soft() -> None:
    """The five split channels must re-sum to the legacy cost_hard / cost_soft."""
    env, _, _, batch = _collect(12)
    cost_hard, cost_soft = ppo.verifier_cost(env, batch, w_carped=3.0)
    ch = ppo.verifier_costs_split(env, batch, w_carped=3.0)

    np.testing.assert_allclose(ch["car_crash"] + ch["ped_hit"], cost_hard, atol=1e-5)
    np.testing.assert_allclose(
        ch["car_risk"] + ch["ped_yield"] + ch["lane"], cost_soft, atol=1e-5,
    )


def test_update_with_explicit_penalty_runs() -> None:
    """ppo.update with a precomputed penalty array returns finite metrics."""
    env, cfg, ts, batch = _collect(13)
    channels = ppo.verifier_costs_split(env, batch, w_carped=3.0)
    lams = {"car_crash": 5.0, "ped_hit": 7.0,
            "car_risk": 2.0, "ped_yield": 2.0, "lane": 1.0}
    penalty = sum(lams[k] * channels[k] for k in channels)

    ts2, metrics = ppo.update(env, cfg, ts, batch, penalty=penalty)
    assert jnp.isfinite(metrics["loss"]), f"loss not finite: {metrics['loss']}"
    assert jnp.isfinite(metrics["ep_reward"]), (
        f"ep_reward not finite: {metrics['ep_reward']}"
    )
    assert ts2 is not ts


def test_penalty_takes_priority_over_lam_args() -> None:
    """A zero penalty must equal the unpenalised reward path (penalty wins)."""
    env, cfg, ts, batch = _collect(14)
    B, T, N = np.asarray(batch["reward"]).shape
    zero_penalty = jnp.zeros((B, T, N), jnp.float32)

    # With penalty=0 and bogus lam args, the lam args must be ignored.
    _, m_pen = ppo.update(env, cfg, ts, batch, lam=999.0, penalty=zero_penalty)
    # Baseline: legacy single-lam with lam=0 (no penalty).
    _, m_base = ppo.update(env, cfg, ts, batch, lam=0.0)

    np.testing.assert_allclose(
        float(m_pen["ep_reward"]), float(m_base["ep_reward"]), atol=1e-4,
    )
