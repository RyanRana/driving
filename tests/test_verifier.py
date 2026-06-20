"""Phase 4 — the deterministic verifier. Same trace -> same verdict; planted
crash / off-road / clean traces each score correctly; and the offline verdict
agrees with the env's online constraint info on a real rollout.
"""
import numpy as np

from smoothride.env.trace import TIMELINE_FIELDS, Manifest, Trace


def _trace(T=10, N=2, **overrides) -> Trace:
    tl = {f: np.zeros((T, N), np.float32) for f in TIMELINE_FIELDS}
    # default: two cars far apart, under the limit, on-road, never arriving
    tl["x"][:, 0] = np.arange(T) * 5.0
    tl["x"][:, 1] = 1000.0 + np.arange(T) * 5.0
    tl["speed"][:] = 8.0
    tl["speed_limit"][:] = 14.0
    for k, v in overrides.items():
        tl[k] = v
    return Trace(manifest=Manifest("test", seed=0), dt=0.2, timeline=tl,
                 collision_radius=2.2)


def test_clean_run_is_valid():
    from smoothride.eval.verifier import verify
    v = verify(_trace())
    assert v.valid_run is True
    assert v.crash_count == 0 and v.offroad_count == 0 and v.rule_count == 0


def test_determinism_same_trace_same_verdict():
    from smoothride.eval.verifier import verify
    tr = _trace()
    a, b = verify(tr).summary(), verify(tr).summary()
    assert a == b


def test_planted_crash_detected():
    from smoothride.eval.verifier import verify
    T, N = 10, 2
    tr = _trace(T=T, N=N)
    # drive car 1 on top of car 0 at step 5 (distance 0 < collision_radius)
    tr.timeline["x"][5, 1] = tr.timeline["x"][5, 0]
    tr.timeline["y"][5, 1] = tr.timeline["y"][5, 0]
    v = verify(tr)
    assert v.crash_count >= 2          # both cars flagged at that step
    assert v.valid_run is False


def test_offroad_from_channel_detected():
    from smoothride.eval.verifier import verify
    off = np.zeros((10, 2), np.float32)
    off[3, 0] = 1.0
    v = verify(_trace(off_road=off))
    assert v.offroad_count == 1 and v.valid_run is False


def test_speeding_is_a_rule_violation():
    from smoothride.eval.verifier import verify
    spd = np.full((10, 2), 8.0, np.float32)
    spd[4, 1] = 20.0   # well over the 14 m/s limit
    v = verify(_trace(speed=spd))
    assert v.rule_count == 1 and v.valid_run is False


def test_arrival_travel_time():
    from smoothride.eval.verifier import verify
    arr = np.zeros((10, 2), np.float32)
    arr[6, 0] = 1.0    # car 0 arrives at step 6 (1-indexed: 7*dt)
    v = verify(_trace(arrived=arr))
    assert v.trips == 1
    assert v.cars[0].arrived and np.isclose(v.cars[0].travel_time, 7 * 0.2)


def test_roundtrip_jsonl(tmp_path):
    from smoothride.env.trace import read_jsonl, write_jsonl
    from smoothride.eval.verifier import verify
    tr = _trace()
    p = write_jsonl(tr, str(tmp_path / "t.jsonl"))
    back = read_jsonl(p)
    assert verify(back).summary() == verify(tr).summary()


def test_verifier_agrees_with_env_online_info():
    """On a real 3D rollout, the offline crash count matches the env's online
    just_crashed events (the verifier reproduces the env's geometric judgment)."""
    import jax
    from smoothride.data.map_loader import load_road_network
    from smoothride.env import kinematic as K
    from smoothride.env.routing import build_route_pool
    from smoothride.env import trace as TR
    from smoothride.eval.verifier import verify

    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=128)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=16, n_peds=4, max_steps=80)
    roll = TR.rollout(env, TR.random_params(env), jax.random.PRNGKey(3))
    tr = TR.build_trace(env, Manifest.from_env(env, "rt", 3), roll)
    v = verify(tr)
    online_crashes = int((roll["crash"] > 0.5).sum())
    # the verifier recomputes crashes geometrically; it should see at least the
    # online collisions (it can also catch overlaps the online immunity masked).
    assert v.crash_count >= online_crashes
    assert v.n_cars == 16
