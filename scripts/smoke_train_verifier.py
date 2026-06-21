"""Full training-loop smoke: the verifier DRIVES PPO.

Each iteration: roll out on device -> the verifier relabels the rollout into a
per-step cost (rl/verifier via ppo.verifier_cost) -> PPO-Lagrangian optimizes
reward_eff = reward - lam*cost. The reward is now efficiency-only (§9), so every
crash/lane/wrong-way signal reaches the policy through the verifier's cost.

Runs locally on CPU for a handful of iterations (training proper runs on Modal).
This is a smoke: it proves the loop wires up and steps without NaNs, not that a few
CPU iterations produce a good policy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from smoothride.data.map_loader import load_road_network
from smoothride.env import kinematic as K
from smoothride.env.routing import build_route_pool
from smoothride.rl import ppo

N_ITERS, N_WORLDS, TARGET_COST = 5, 8, 0.05


def main() -> None:
    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=256)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=16, n_peds=8, max_steps=60)
    cfg = ppo.PPOConfig(n_worlds=N_WORLDS, epochs=2, minibatches=4)

    key = jax.random.PRNGKey(0)
    key, kinit = jax.random.split(key)
    ts = ppo.make_train_state(env, cfg, kinit)
    lam = 0.0  # Lagrangian multiplier, dual-ascended toward TARGET_COST

    print(f"env: agents={env.n_agents} steps={env.max_steps} worlds={N_WORLDS}")
    print(f"reward is efficiency-only (w_progress={env.w_progress} "
          f"w_goal={env.w_goal} w_time={env.w_time}); cost = verifier signal\n")

    for it in range(N_ITERS):
        key, kc = jax.random.split(key)
        batch = ppo.collect(env, ts, kc, cfg.n_worlds)

        # --- the verifier produces the training signal ---
        crash_only = float(jnp.asarray(batch["cost"]).mean())   # old in-loop cost
        v_cost = ppo.verifier_cost(env, batch)                   # widened, by verifier
        batch = {**batch, "cost": v_cost}
        mean_cost = float(v_cost.mean())

        ts, m = ppo.update(env, cfg, ts, batch, lam=lam)
        lam = max(0.0, lam + 0.5 * (mean_cost - TARGET_COST))    # dual ascent

        print(f"iter {it}: loss={float(m['loss']):+.3f} "
              f"ep_reward={float(m['ep_reward']):+.2f} "
              f"verifier_cost={mean_cost:.3f} (crash_only={crash_only:.3f}) "
              f"lam={lam:.3f} goals/agent={float(m['goals_per_agent']):.2f}")

    assert np.isfinite(float(m["loss"])), "loss went non-finite"
    print("\nOK — verifier-driven PPO ran end to end (cost = verifier, reward = §9).")


if __name__ == "__main__":
    main()
