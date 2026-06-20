"""Phase 2 — grade-aware dynamics. Closed-form checks on flat ground still hold,
and an uphill grade demonstrably saps achieved speed for an identical action.

We synthesize a single straight route so the kinematics have an analytic answer
(no dependence on the real SF graph geometry).
"""
import jax
import jax.numpy as jnp
import numpy as np

from smoothride.env import kinematic as K
from smoothride.env.routing import RoutePool


def _straight_pool(grade: float = 0.0, length: float = 4000.0) -> RoutePool:
    """One long straight eastbound route, optionally on a constant grade."""
    W = 8
    xs = np.linspace(0, length, W, dtype=np.float32)
    xy = np.stack([xs, np.full(W, 50.0, np.float32)], axis=1)[None]   # (1,W,2)
    z = (xs * grade)[None].astype(np.float32)
    return RoutePool(
        xy=xy, n=np.array([W], np.int32),
        node=np.zeros((1, W), np.int32), junc=np.zeros((1, W), bool),
        lanes=np.ones((1, W), np.int32),
        speed=np.full((1, W), 100.0, np.float32),   # high cap: don't clamp
        z=z, grade=np.full((1, W), grade, np.float32),
    )


def _env(pool, **kw):
    return K.make_env(pool, (-10.0, 0.0), (4100.0, 100.0),
                      cell_size=200.0, n_agents=1, n_peds=1, max_steps=200, **kw)


def test_flat_constant_accel_matches_kinematics():
    """Full-throttle from rest on flat ground: v = a*t, s = 1/2 a t^2."""
    env = _env(_straight_pool(grade=0.0))
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    st = st.replace(speed=jnp.zeros(1), wp_ptr=jnp.array([1]))
    x0 = float(st.pos[0, 0])
    step = jax.jit(lambda s, a, k: K.step(env, s, a, k))
    act = jnp.array([[1.0, 0.0, 0.0]])  # max accel, no steer
    n = 10
    for _ in range(n):
        st, *_ = step(st, act, jax.random.PRNGKey(0))
    t = n * env.dt
    assert np.isclose(float(st.speed[0]), env.accel_max * t, rtol=1e-4)
    # distance: discrete forward-Euler sum_{i=1..n} a*(i*dt)*dt
    exp_dist = env.accel_max * env.dt * env.dt * sum(range(1, n + 1))
    assert np.isclose(float(st.pos[0, 0]) - x0, exp_dist, rtol=1e-3)


def test_speed_clamped_to_vmax():
    env = _env(_straight_pool(grade=0.0))
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    st = st.replace(speed=jnp.zeros(1), wp_ptr=jnp.array([1]))
    step = jax.jit(lambda s, a, k: K.step(env, s, a, k))
    act = jnp.array([[1.0, 0.0, 0.0]])
    for _ in range(200):
        st, *_ = step(st, act, jax.random.PRNGKey(0))
    assert float(st.speed[0]) <= env.v_max + 1e-4


def test_uphill_reduces_achieved_speed():
    """Same full-throttle action: a steep uphill ends slower than flat ground."""
    act = jnp.array([[1.0, 0.0, 0.0]])
    speeds = {}
    for name, grade in [("flat", 0.0), ("uphill", 0.25)]:
        env = _env(_straight_pool(grade=grade))
        st, _ = K.reset(env, jax.random.PRNGKey(0))
        st = st.replace(speed=jnp.zeros(1), wp_ptr=jnp.array([1]),
                        ped_pos=jnp.array([[1e6, 1e6]]))
        step = jax.jit(lambda s, a, k: K.step(env, s, a, k))
        for _ in range(8):
            st, *_ = step(st, act, jax.random.PRNGKey(0))
        speeds[name] = float(st.speed[0])
    assert speeds["uphill"] < speeds["flat"]
