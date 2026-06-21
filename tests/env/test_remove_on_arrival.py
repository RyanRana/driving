"""Finite-cohort remove-on-arrival: a car that reaches its destination FREEZES
at the destination, is MASKED OUT of collision/reward, and NEVER respawns. Its
arrival is latched so the verifier/viewer can mark it "completed" (green)."""
import jax
import jax.numpy as jnp
import numpy as np

from smoothride.env import kinematic as K
from smoothride.env.routing import RoutePool


def _straight_pool():
    # one route, 3 waypoints along +x: (0,0)-(50,0)-(100,0)
    xy = np.array([[[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]], np.float32)  # (1,3,2)
    return RoutePool(
        xy=xy,
        n=np.array([3], np.int32),
        node=np.array([[0, 1, 2]], np.int32),
        junc=np.zeros((1, 3), bool),
        lanes=np.ones((1, 3), np.int32),
        speed=np.full((1, 3), 16.0, np.float32),
    )


def _env(n_agents=2):
    return K.make_env(_straight_pool(), world_min=[-20.0, -20.0],
                      world_max=[120.0, 120.0], n_agents=n_agents, n_peds=1,
                      max_steps=50)


def _arrived_state(env):
    """A state where car 0 sits on its final waypoint (about to be flagged arrived)
    and car 1 is parked far away, mid-route. The lone pedestrian is parked in a
    far corner so it never interferes."""
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    pos = np.array(st.pos)
    pos[0] = [100.0, 0.0]      # on the final waypoint
    pos[1] = [50.0, 60.0]      # well clear of car 0
    return st.replace(
        pos=jnp.asarray(pos, jnp.float32),
        speed=st.speed.at[0].set(0.0).at[1].set(0.0),
        wp_ptr=st.wp_ptr.at[0].set(2).at[1].set(1),
        spawn_grace=jnp.zeros_like(st.spawn_grace),   # clear grace -> behavior testable
        ped_pos=jnp.array([[10.0, 110.0]], jnp.float32),
    )


def test_reaching_destination_latches_arrived():
    env = _env()
    st = _arrived_state(env)
    nst, _, _, _, info = K.step(env, st, jnp.zeros((2, 3)), jax.random.PRNGKey(1))
    assert bool(nst.arrived[0]) is True
    assert bool(info["arrived"][0]) is True
    assert bool(nst.arrived[1]) is False


def test_arrived_car_freezes_and_never_respawns():
    env = _env()
    st = _arrived_state(env)
    nst, *_ = K.step(env, st, jnp.zeros((2, 3)), jax.random.PRNGKey(1))
    frozen_pos = np.array(nst.pos[0])
    route0 = int(nst.route_idx[0])
    # drive several more steps with full-throttle action; the arrived car must NOT move
    s = nst
    for k in range(5):
        s, *_ = K.step(env, s, jnp.ones((2, 3)), jax.random.PRNGKey(10 + k))
        assert np.allclose(np.array(s.pos[0]), frozen_pos, atol=1e-4)  # frozen in place
        assert float(s.speed[0]) == 0.0                                # stopped
        assert int(s.route_idx[0]) == route0                           # NO respawn
        assert bool(s.arrived[0]) is True                              # latch persists


def test_arrived_car_is_masked_out_of_collisions():
    env = _env()
    st = _arrived_state(env)
    nst, *_ = K.step(env, st, jnp.zeros((2, 3)), jax.random.PRNGKey(1))
    # drop the active car 1 directly on top of the frozen, arrived car 0
    nst = nst.replace(pos=nst.pos.at[1].set(nst.pos[0]),
                      spawn_grace=jnp.zeros_like(nst.spawn_grace))
    _, _, _, _, info = K.step(env, nst, jnp.zeros((2, 3)), jax.random.PRNGKey(2))
    assert bool(info["just_crashed"][1]) is False   # frozen car is not an obstacle
    assert bool(info["just_crashed"][0]) is False   # done car never re-crashes


def test_arrived_car_earns_zero_reward_after_arrival():
    env = _env()
    st = _arrived_state(env)
    nst, *_ = K.step(env, st, jnp.zeros((2, 3)), jax.random.PRNGKey(1))   # car0 arrives here
    _, _, reward, _, _ = K.step(env, nst, jnp.ones((2, 3)), jax.random.PRNGKey(2))
    assert float(reward[0]) == 0.0   # already done -> no further reward/penalty


def test_active_car_still_moves():
    env = _env()
    st = _arrived_state(env)
    # car 1 is active mid-route; a forward action should move it
    before = np.array(st.pos[1])
    nst, *_ = K.step(env, st, jnp.ones((2, 3)), jax.random.PRNGKey(3))
    assert not np.allclose(np.array(nst.pos[1]), before)
    assert bool(nst.arrived[1]) is False
