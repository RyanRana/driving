"""Reproducible run traces — the bridge from a rollout to the verifier + viewer.

A trace is everything you need to (a) replay the run bit-for-bit and (b) judge it
WITHOUT re-simulating: a Manifest (the four ids that pin a run) plus a per-car,
per-step Timeline of poses/kinematics/route-progress, plus discrete Events
(crash / off-road / rule / arrival / respawn).

The trace is the ONLY input to `eval.verifier` — same trace, same verdict.
Serialized as JSONL under `runs/traces/<run_id>.jsonl`:
  line 0      : {"kind":"manifest", ...}
  line 1      : {"kind":"meta", dt, n_steps, n_cars, fields...}
  next N lines: {"kind":"car", "id":i, "t":[...], "x":[...], ...}
  last line   : {"kind":"summary", ...}  (filled by the verifier)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import kinematic as K

# the per-car timeline channels written to the trace (and read by the verifier)
TIMELINE_FIELDS = ("x", "y", "z", "heading", "speed", "steer", "lane",
                   "accel", "lane_cmd", "wp_ptr", "dist_remaining",
                   "speed_limit", "crash", "arrived", "off_road")


@dataclass
class Manifest:
    run_id: str
    seed: int
    scenario_id: str = "downtown_sf"
    policy_checkpoint_id: str = "random"
    config_hash: str = ""

    @staticmethod
    def from_env(env: K.Env, run_id: str, seed: int, **kw) -> "Manifest":
        # config_hash pins the env params + map version so a run replays exactly.
        params = {k: getattr(env, k) for k in (
            "n_agents", "n_peds", "k_neighbors", "max_steps", "dt", "v_max",
            "accel_max", "steer_max", "wheelbase", "grade_accel", "lane_width",
            "collision_radius")}
        params["n_routes"] = int(env.routes_xy.shape[0])
        params["n_bld_segs"] = int(env.bld_segs.shape[0])
        h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return Manifest(run_id=run_id, seed=seed, config_hash=h[:16], **kw)


@dataclass
class Trace:
    manifest: Manifest
    dt: float
    timeline: dict[str, np.ndarray]   # each (T, N)
    collision_radius: float
    # destination waypoint count per car-route, for the arrival predicate
    route_n: np.ndarray = field(default=None)

    @property
    def n_steps(self) -> int:
        return self.timeline["x"].shape[0]

    @property
    def n_cars(self) -> int:
        return self.timeline["x"].shape[1]


def _trace_record(env: K.Env, nst: K.State, action: jnp.ndarray, info: dict) -> dict:
    """Per-step timeline channels (one row of the trace) from a stepped state."""
    n = env.routes_n[nst.route_idx]
    cum = env.routes_cum[nst.route_idx]
    idx = jnp.arange(env.n_agents)
    dist_rem = cum[idx, jnp.maximum(n - 1, 0)] - cum[idx, nst.wp_ptr]
    speed_lim = jnp.minimum(env.v_max, env.routes_speed[nst.route_idx, nst.wp_ptr])
    return {
        "x": nst.pos[:, 0], "y": nst.pos[:, 1], "z": nst.z,
        "heading": nst.heading, "speed": nst.speed,
        "steer": jnp.clip(action[:, 1], -1, 1) * env.steer_max,
        "lane": nst.lane.astype(jnp.float32),
        "accel": jnp.clip(action[:, 0], -1, 1),
        "lane_cmd": action[:, 2],
        "wp_ptr": nst.wp_ptr.astype(jnp.float32),
        "dist_remaining": dist_rem,
        "speed_limit": speed_lim,
        "crash": info["just_crashed"].astype(jnp.float32),
        "arrived": info["arrived"].astype(jnp.float32),
        "off_road": info["cost_offroad"],
    }


def rollout(env: K.Env, params, key, sample: bool = True) -> dict:
    """Replay one world under `params`; return per-step arrays for the trace.

    `params` may be real (loaded) or freshly-initialized (random policy) — the 3D
    geometry/terrain/occlusion is exercised either way.
    """
    from ..rl.networks import ActorCritic
    net = ActorCritic(act_dim=env.act_dim)

    def step_fn(carry, k):
        st, obs = carry
        gf = jnp.broadcast_to(obs.mean(-2, keepdims=True), obs.shape)
        mean, log_std, _ = net.apply(params, obs, gf)
        ka, kn = jax.random.split(k)
        action = mean + (jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
                         if sample else 0.0)
        nst, nobs, r, done, info = K.step(env, st, action, kn)
        return (nst, nobs), _trace_record(env, nst, action, info)

    kr, ks = jax.random.split(key)
    st, obs = K.reset(env, kr)
    keys = jax.random.split(ks, env.max_steps)
    _, recs = jax.lax.scan(step_fn, (st, obs), keys)
    return {k: np.asarray(v) for k, v in recs.items()}


def _heuristic_action(env: K.Env, st: K.State) -> jnp.ndarray:
    """A scripted waypoint-follower: steer toward the lane target, cruise toward
    the speed limit, ease off when a lead car is close. Lets the demo show cars
    actually driving the SF streets without a trained checkpoint."""
    tgt = K._target_wp(env, st)
    to_wp = tgt - st.pos
    herr = K._wrap(jnp.arctan2(to_wp[:, 1], to_wp[:, 0]) - st.heading)
    steer = jnp.clip(herr / env.steer_max, -1.0, 1.0)
    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])
    # gentle car-following: accelerate unless already at the local speed cap
    accel = jnp.where(st.speed < 0.9 * vmax, 0.7, -0.2)
    lane_cmd = jnp.zeros_like(accel)
    return jnp.stack([accel, steer, lane_cmd], axis=-1)


def rollout_heuristic(env: K.Env, key) -> dict:
    """Same per-step trace as `rollout`, driven by the scripted controller."""
    def step_fn(carry, k):
        st, _ = carry
        action = _heuristic_action(env, st)
        nst, nobs, r, done, info = K.step(env, st, action, k)
        rec = _trace_record(env, nst, action, info)
        return (nst, nobs), rec

    kr, ks = jax.random.split(key)
    st, obs = K.reset(env, kr)
    keys = jax.random.split(ks, env.max_steps)
    _, recs = jax.lax.scan(step_fn, (st, obs), keys)
    return {k: np.asarray(v) for k, v in recs.items()}


def build_trace(env: K.Env, manifest: Manifest, roll: dict) -> Trace:
    timeline = {k: roll[k] for k in TIMELINE_FIELDS}
    return Trace(manifest=manifest, dt=float(env.dt), timeline=timeline,
                 collision_radius=float(env.collision_radius))


def write_jsonl(trace: Trace, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    T, N = trace.n_steps, trace.n_cars
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "manifest",
                            **dataclasses.asdict(trace.manifest)}) + "\n")
        f.write(json.dumps({"kind": "meta", "dt": trace.dt, "n_steps": T,
                            "n_cars": N, "collision_radius": trace.collision_radius,
                            "fields": list(TIMELINE_FIELDS)}) + "\n")
        for i in range(N):
            rec = {"kind": "car", "id": i}
            for fld in TIMELINE_FIELDS:
                col = trace.timeline[fld][:, i]
                rnd = 4 if fld in ("x", "y", "z", "heading", "speed", "steer",
                                   "dist_remaining", "speed_limit", "accel",
                                   "lane_cmd") else 0
                rec[fld] = [round(float(v), rnd) for v in col]
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return path


def read_jsonl(path: str) -> Trace:
    manifest = meta = None
    cols: dict[str, list] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["kind"] == "manifest":
                obj.pop("kind")
                manifest = Manifest(**obj)
            elif obj["kind"] == "meta":
                meta = obj
            elif obj["kind"] == "car":
                for fld in meta["fields"]:
                    cols.setdefault(fld, []).append(obj[fld])
    timeline = {fld: np.asarray(v, np.float32).T for fld, v in cols.items()}
    return Trace(manifest=manifest, dt=meta["dt"], timeline=timeline,
                 collision_radius=meta["collision_radius"])


def random_params(env: K.Env, seed: int = 0):
    """Freshly-initialized policy params (used when no checkpoint exists)."""
    from ..rl import ppo
    ts = ppo.make_train_state(env, ppo.PPOConfig(), jax.random.PRNGKey(seed))
    return ts.params
