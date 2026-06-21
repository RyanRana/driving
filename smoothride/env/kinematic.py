"""Vectorized kinematic multi-agent driving environment (JAX) — v3 (scales).

The behavioral layer of SmoothRide. Cars follow routes on the real SF graph under
a kinematic-bicycle model and learn to coordinate, at realistic city density.

Models the actual environment:
  * per-edge SPEED LIMITS (highways fast, surface slow)
  * MULTI-LANE roads + lane-change action
  * lead-vehicle GAP -> car-following + stop-and-go
  * INTERSECTION yielding (right-of-way without traffic lights)
  * unruly PEDESTRIANS to avoid (scored as a constraint, not reward)
  * FINITE COHORT: each car runs one trip then freezes (arrive/crash), no respawn
  * NON-OVERLAPPING spawns: cars/peds never start within a collision (root-cause)
  * CMDP reward (§9): efficiency only (progress + arrival − time); crash/lane/
    proximity constraints flow through the deterministic verifier's cost channel
  * SPATIAL-HASH neighbor search -> O(N*C), scales to thousands of cars

Pure `reset`/`step` over one world of N cars + M peds; `vmap` over worlds.
"""
from __future__ import annotations

import flax.struct as struct
import jax
import jax.numpy as jnp

from . import spatial
from .routing import RoutePool


@struct.dataclass
class Env:
    routes_xy: jnp.ndarray
    routes_n: jnp.ndarray
    routes_node: jnp.ndarray
    routes_junc: jnp.ndarray
    routes_lanes: jnp.ndarray
    routes_speed: jnp.ndarray
    routes_cum: jnp.ndarray     # (P, W) cumulative arc length (for spread spawning)
    world_min: jnp.ndarray
    world_max: jnp.ndarray

    n_agents: int = struct.field(pytree_node=False, default=24)
    n_peds: int = struct.field(pytree_node=False, default=12)
    k_neighbors: int = struct.field(pytree_node=False, default=4)
    max_steps: int = struct.field(pytree_node=False, default=300)
    cell_size: float = struct.field(pytree_node=False, default=35.0)
    cap: int = struct.field(pytree_node=False, default=16)
    ncx: int = struct.field(pytree_node=False, default=1)
    ncy: int = struct.field(pytree_node=False, default=1)
    cand_C: int = struct.field(pytree_node=False, default=144)
    spread_spawn: bool = struct.field(pytree_node=False, default=True)
    spawn_sep: float = struct.field(pytree_node=False, default=6.0)    # min car-car spawn gap, m
    spawn_tries: int = struct.field(pytree_node=False, default=12)     # reject-sampling rounds
    dt: float = struct.field(pytree_node=False, default=0.2)
    v_max: float = struct.field(pytree_node=False, default=16.0)
    accel_max: float = struct.field(pytree_node=False, default=3.0)
    steer_max: float = struct.field(pytree_node=False, default=0.5)
    wheelbase: float = struct.field(pytree_node=False, default=2.7)
    lane_width: float = struct.field(pytree_node=False, default=3.5)
    wp_radius: float = struct.field(pytree_node=False, default=9.0)
    # collision_radius MUST be < lane_width, else adjacent-lane cars "collide"
    collision_radius: float = struct.field(pytree_node=False, default=2.2)
    ped_radius: float = struct.field(pytree_node=False, default=2.2)
    ped_speed: float = struct.field(pytree_node=False, default=1.4)
    prox_radius: float = struct.field(pytree_node=False, default=6.0)
    lead_cone: float = struct.field(pytree_node=False, default=30.0)
    junc_zone: float = struct.field(pytree_node=False, default=14.0)
    idle_speed: float = struct.field(pytree_node=False, default=0.5)
    w_progress: float = struct.field(pytree_node=False, default=1.0)
    w_idle: float = struct.field(pytree_node=False, default=0.05)
    w_prox: float = struct.field(pytree_node=False, default=1.0)
    w_collision: float = struct.field(pytree_node=False, default=40.0)
    w_ped_prox: float = struct.field(pytree_node=False, default=2.0)
    w_ped: float = struct.field(pytree_node=False, default=80.0)
    w_yield: float = struct.field(pytree_node=False, default=1.5)
    w_goal: float = struct.field(pytree_node=False, default=15.0)
    # CMDP reframe (§9): reward is efficiency only (progress + arrival − time); all
    # crash/lane/proximity constraints leave the reward and go through the verifier's
    # cost channel. w_idle/w_prox/w_collision/w_ped*/w_yield are retained for config
    # compatibility but no longer shape the reward.
    w_time: float = struct.field(pytree_node=False, default=0.02)

    @property
    def obs_dim(self) -> int:
        return 6 + 1 + self.k_neighbors * 4 + 3

    @property
    def act_dim(self) -> int:
        return 3


@struct.dataclass
class State:
    pos: jnp.ndarray
    heading: jnp.ndarray
    speed: jnp.ndarray
    route_idx: jnp.ndarray
    wp_ptr: jnp.ndarray
    lane: jnp.ndarray
    just_crashed: jnp.ndarray  # (N,) bool: collided THIS step (terminal: car freezes)
    crashes: jnp.ndarray       # (N,) int: cumulative collisions
    spawn_grace: jnp.ndarray   # (N,) int: countdown of merge-in immunity steps
    arrived: jnp.ndarray       # (N,) bool: reached destination (LATCHES; car freezes)
    goals: jnp.ndarray
    ped_pos: jnp.ndarray
    ped_dir: jnp.ndarray
    t: jnp.ndarray


SPAWN_GRACE = 4   # steps of collision immunity after entering the map


def _wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _route_dir(env: Env, route_idx, wp_ptr):
    n = env.routes_n[route_idx]
    cur = env.routes_xy[route_idx, wp_ptr]
    nxt = env.routes_xy[route_idx, jnp.minimum(wp_ptr + 1, n - 1)]
    d = nxt - cur
    return d / (jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-6), cur


def _target_wp(env: Env, st: State) -> jnp.ndarray:
    dn, cur = _route_dir(env, st.route_idx, st.wp_ptr)
    right = jnp.stack([dn[..., 1], -dn[..., 0]], axis=-1)
    offset = env.lane_width * (st.lane.astype(jnp.float32) + 0.5)
    return cur + right * offset[..., None]


def _spawn(env: Env, idx):
    dn, cur = _route_dir(env, idx, jnp.ones_like(idx))
    return cur, jnp.arctan2(dn[..., 1], dn[..., 0])


def _spread_spawn(env: Env, idx, key):
    """Spawn at a random fraction ALONG the route -> spread across the network."""
    nwp = env.routes_n[idx]
    frac = jax.random.uniform(key, idx.shape)
    wp = jnp.clip((frac * (nwp - 1)).astype(jnp.int32), 1, jnp.maximum(nwp - 1, 1))
    dn, cur = _route_dir(env, idx, wp)
    return cur, jnp.arctan2(dn[..., 1], dn[..., 0]), wp


def _entry_spawn(env: Env, idx, key):
    """spread_spawn=True: spread along route (city). False: enter at the route
    START (boundary edge) and drive THROUGH the junction (scenario windows)."""
    if env.spread_spawn:
        return _spread_spawn(env, idx, key)
    dn, cur = _route_dir(env, idx, jnp.ones_like(idx))
    return cur, jnp.arctan2(dn[..., 1], dn[..., 0]), jnp.ones_like(idx, jnp.int32)


def _candidates(env: Env, pos):
    return spatial.grid_candidates(pos, env.world_min, env.cell_size,
                                   env.ncx, env.ncy, env.cap, env.cand_C)


def _observe(env: Env, st: State, cand) -> jnp.ndarray:
    N = env.n_agents
    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])
    tgt = _target_wp(env, st)
    to_wp = tgt - st.pos
    dist = jnp.linalg.norm(to_wp, axis=-1)
    herr = _wrap(jnp.arctan2(to_wp[:, 1], to_wp[:, 0]) - st.heading)
    n = env.routes_n[st.route_idx]
    progress = st.wp_ptr / jnp.maximum(n - 1, 1)
    fdir = jnp.stack([jnp.cos(st.heading), jnp.sin(st.heading)], -1)

    # candidate relative geometry (N, C, 2)
    valid = (cand >= 0) & (cand != jnp.arange(N)[:, None])
    rel = st.pos[cand] - st.pos[:, None, :]
    cd = jnp.where(valid, jnp.linalg.norm(rel, axis=-1), 1e9)
    forward = jnp.sum(rel * fdir[:, None, :], -1)
    lateral = jnp.abs(rel[..., 0] * fdir[:, None, 1] - rel[..., 1] * fdir[:, None, 0])
    ahead = valid & (forward > 0.5) & (lateral < env.lane_width) & (forward < env.lead_cone)
    lead_gap = jnp.min(jnp.where(ahead, forward, env.lead_cone), axis=-1)

    lane_frac = st.lane.astype(jnp.float32) / jnp.maximum(
        env.routes_lanes[st.route_idx, st.wp_ptr] - 1, 1)
    ego = jnp.stack([
        st.speed / jnp.maximum(vmax, 1.0),
        jnp.sin(herr), jnp.cos(herr),
        jnp.clip(dist / 100.0, 0, 1), progress,
        jnp.clip(lead_gap / env.lead_cone, 0, 1),
    ], axis=-1)

    # K nearest among candidates (ego frame)
    _, kk = jax.lax.top_k(-cd, env.k_neighbors)             # (N,K) into candidate axis
    nbr = jnp.take_along_axis(cand, kk, axis=1)             # (N,K) agent idx
    nbr_valid = jnp.take_along_axis(valid, kk, axis=1)
    c, s = jnp.cos(-st.heading), jnp.sin(-st.heading)
    nrel = st.pos[nbr] - st.pos[:, None, :]
    nx = (nrel[..., 0] * c[:, None] - nrel[..., 1] * s[:, None]) * nbr_valid
    ny = (nrel[..., 0] * s[:, None] + nrel[..., 1] * c[:, None]) * nbr_valid
    vel = st.speed[:, None] * fdir
    nvel = (vel[nbr] - vel[:, None, :]) * nbr_valid[..., None]
    nbr_feat = jnp.concatenate([
        jnp.clip(nx[..., None] / 50, -1, 1),
        jnp.clip(ny[..., None] / 50, -1, 1),
        jnp.clip(nvel / env.v_max, -1, 1),
    ], -1).reshape(N, -1)

    # nearest pedestrian (peds are few -> brute force)
    pd = st.pos[:, None, :] - st.ped_pos[None, :, :]
    pdist = jnp.linalg.norm(pd, axis=-1)
    pj = jnp.argmin(pdist, axis=-1)
    prel = st.ped_pos[pj] - st.pos
    px = prel[:, 0] * c - prel[:, 1] * s
    py = prel[:, 0] * s + prel[:, 1] * c
    pmin = pdist[jnp.arange(N), pj]
    ped_feat = jnp.stack([jnp.clip(px / 50, -1, 1), jnp.clip(py / 50, -1, 1),
                          jnp.clip(pmin / 50, 0, 1)], -1)

    return jnp.concatenate([ego, lane_frac[:, None], nbr_feat, ped_feat], -1)


def _ped_step(env: Env, st: State, key):
    kd, kj = jax.random.split(key)
    turn = jax.random.normal(kd, (env.n_peds,)) * 0.3
    dart = (jax.random.uniform(kj, (env.n_peds,)) < 0.02).astype(jnp.float32)
    ped_dir = _wrap(st.ped_dir + turn + dart * jax.random.normal(kd, (env.n_peds,)))
    step = env.ped_speed * env.dt * jnp.stack([jnp.cos(ped_dir), jnp.sin(ped_dir)], -1)
    ped_pos = jnp.clip(st.ped_pos + step, env.world_min, env.world_max)
    flip = (st.ped_pos + step < env.world_min) | (st.ped_pos + step > env.world_max)
    ped_dir = jnp.where(flip[:, 0], jnp.pi - ped_dir, ped_dir)
    ped_dir = jnp.where(flip[:, 1], -ped_dir, ped_dir)
    return ped_pos, _wrap(ped_dir)


def _place_cars(env: Env, key: jax.Array):
    """Spawn cars on routes with NO two within env.spawn_sep. Bounded reject-sampling:
    each round, any car closer than spawn_sep to a lower-indexed car is re-spawned on
    a fresh random route/slot. Converges for feasible densities; spawn_grace covers any
    rare residual so a leftover near-miss still isn't counted as a crash."""
    n = env.n_agents
    n_routes = env.routes_xy.shape[0]
    idx = jnp.arange(n)
    lower = idx[:, None] > idx[None, :]      # (n,n) True where m < j

    def sample(k):
        ka, kb = jax.random.split(k)
        ridx = jax.random.randint(ka, (n,), 0, n_routes)
        pos, head, wp = _entry_spawn(env, ridx, kb)
        return ridx, pos, head, wp

    def body(_, carry):
        key, ridx, pos, head, wp = carry
        d = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        conflict = jnp.any((d < env.spawn_sep) & lower, axis=1)        # move the later car
        key, ks = jax.random.split(key)
        ridx2, pos2, head2, wp2 = sample(ks)
        ridx = jnp.where(conflict, ridx2, ridx)
        pos = jnp.where(conflict[:, None], pos2, pos)
        head = jnp.where(conflict, head2, head)
        wp = jnp.where(conflict, wp2, wp)
        return key, ridx, pos, head, wp

    key, k0 = jax.random.split(key)
    carry = (key, *sample(k0))
    _, ridx, pos, head, wp = jax.lax.fori_loop(0, env.spawn_tries, body, carry)
    return ridx, pos, head, wp


def _place_peds(env: Env, key: jax.Array, car_pos):
    """Place pedestrians uniformly but never within (ped+collision) radius of a car."""
    m = env.n_peds
    sep = env.ped_radius + env.collision_radius

    def sample(k):
        return jax.random.uniform(k, (m, 2), minval=env.world_min, maxval=env.world_max)

    def body(_, carry):
        key, ped = carry
        d = jnp.linalg.norm(ped[:, None, :] - car_pos[None, :, :], axis=-1)
        conflict = jnp.any(d < sep, axis=1)
        key, ks = jax.random.split(key)
        ped = jnp.where(conflict[:, None], sample(ks), ped)
        return key, ped

    key, k0 = jax.random.split(key)
    _, ped = jax.lax.fori_loop(0, env.spawn_tries, body, (key, sample(k0)))
    return ped


def reset(env: Env, key: jax.Array):
    kc, kp, kpd = jax.random.split(key, 3)
    n = env.n_agents
    route_idx, pos, heading, wp = _place_cars(env, kc)
    ped_pos = _place_peds(env, kpd, pos)
    st = State(
        pos=pos, heading=heading, speed=jnp.zeros(n),
        route_idx=route_idx, wp_ptr=wp, lane=jnp.zeros(n, jnp.int32),
        just_crashed=jnp.zeros(n, bool), crashes=jnp.zeros(n, jnp.int32),
        # initial merge-in immunity so simultaneous spawns aren't born "crashed"
        # (cars spread apart before grace expires). Minimal spawn-clean for the
        # finite cohort; see docs/HANDOFF-sim-contract.md §0.
        spawn_grace=jnp.full(n, SPAWN_GRACE, jnp.int32),
        arrived=jnp.zeros(n, bool), goals=jnp.zeros(n, jnp.int32),
        ped_pos=ped_pos,
        ped_dir=jax.random.uniform(kp, (env.n_peds,), minval=-jnp.pi, maxval=jnp.pi),
        t=jnp.array(0, jnp.int32),
    )
    return st, _observe(env, st, _candidates(env, st.pos))


def step(env: Env, st: State, action: jnp.ndarray, key: jax.Array):
    N = env.n_agents
    _, _, kped = jax.random.split(key, 3)   # (respawn RNG retired; peds still need a key)

    accel = jnp.clip(action[:, 0], -1, 1) * env.accel_max
    delta = jnp.clip(action[:, 1], -1, 1) * env.steer_max
    lane_cmd = action[:, 2]

    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])
    speed = jnp.clip(st.speed + accel * env.dt, 0.0, vmax)
    heading = _wrap(st.heading + (speed / env.wheelbase) * jnp.tan(delta) * env.dt)
    pos = st.pos + speed[:, None] * jnp.stack([jnp.cos(heading), jnp.sin(heading)], -1) * env.dt

    L = env.routes_lanes[st.route_idx, st.wp_ptr]
    shift = jnp.where(lane_cmd > 0.5, 1, jnp.where(lane_cmd < -0.5, -1, 0))
    lane = jnp.clip(st.lane + shift, 0, L - 1)

    nst0 = st.replace(pos=pos, heading=heading, lane=lane)
    tgt = _target_wp(env, nst0)
    to_wp = tgt - pos
    dist = jnp.linalg.norm(to_wp, axis=-1)
    herr = _wrap(jnp.arctan2(to_wp[:, 1], to_wp[:, 0]) - heading)
    progress = speed * jnp.cos(herr) * env.dt

    # done = this car has already finished (arrived) or crashed -> it is FROZEN at
    # its final spot and removed from the simulation (no respawn, no collisions, no
    # reward). The cohort is finite: each car runs ONE trip, then parks. See §0②.
    done = st.arrived | (st.crashes > 0)

    n_wp = env.routes_n[st.route_idx]
    hit = dist < env.wp_radius
    new_goal = hit & (st.wp_ptr >= n_wp - 1) & ~done
    advance = hit & (st.wp_ptr < n_wp - 1) & ~done
    wp_ptr = jnp.where(advance, st.wp_ptr + 1, st.wp_ptr)

    # spatial-hash neighbor reductions
    cand = _candidates(env, pos)
    valid = (cand >= 0) & (cand != jnp.arange(N)[:, None])
    rel = pos[cand] - pos[:, None, :]
    # EXCLUDE spawn-immune and already-done cars as collision partners/victims: a
    # car merging in, or a frozen finished/crashed car, is never an obstacle and
    # never counts as a crash for either side. Only genuine contact between two
    # active, established cars registers.
    immune = (st.spawn_grace > 0) | done
    neighbor_ok = valid & ~immune[cand]
    cd = jnp.where(neighbor_ok, jnp.linalg.norm(rel, axis=-1), 1e9)
    min_d = cd.min(1)
    car_crash = (min_d < env.collision_radius) & ~immune

    # pedestrians (severe)
    pd = jnp.linalg.norm(pos[:, None, :] - st.ped_pos[None, :, :], axis=-1)
    ped_min = pd.min(axis=-1)
    ped_hit = (ped_min < env.ped_radius) & ~immune
    crash_event = car_crash | ped_hit

    # NO respawn (finite cohort, ours). A car becomes done the step it arrives or
    # crashes; from then on it is frozen at that spot. `done` (above) was the state at
    # the START of this step; cars finishing THIS step keep the position where they
    # finished, then stop. Intersection-yield / proximity penalties are gone from the
    # reward (CMDP §9) — the verifier scores those constraints from the trace.
    arrived = st.arrived | new_goal
    crashes = st.crashes + crash_event.astype(jnp.int32)
    goals = st.goals + new_goal.astype(jnp.int32)
    done_after = arrived | (crashes > 0)

    # freeze: already-done cars hold their pose; cars done as of this step stop moving.
    pos = jnp.where(done[:, None], st.pos, pos)
    heading = jnp.where(done, st.heading, heading)
    wp_ptr = jnp.where(done, st.wp_ptr, wp_ptr)
    lane = jnp.where(done, st.lane, lane)
    speed = jnp.where(done_after, 0.0, speed)
    spawn_grace = jnp.maximum(st.spawn_grace - 1, 0)
    ped_pos, ped_dir = _ped_step(env, st, kped)

    # CMDP reward (§9): efficiency only — progress along route, arrival bonus, and a
    # small per-step time cost. Crash/lane/proximity constraints are scored by the
    # verifier's cost channel (rl/verifier.cost_signal), never folded in here.
    reward = (env.w_progress * progress
              + env.w_goal * new_goal.astype(jnp.float32)
              - env.w_time)
    # a car already finished at the start of the step earns nothing further.
    reward = jnp.where(done, 0.0, reward)

    t = st.t + 1
    nst = State(pos=pos, heading=heading, speed=speed, route_idx=st.route_idx,
                wp_ptr=wp_ptr, lane=lane, just_crashed=crash_event, crashes=crashes,
                spawn_grace=spawn_grace, arrived=arrived,
                goals=goals, ped_pos=ped_pos, ped_dir=ped_dir, t=t)
    info = {"just_crashed": crash_event, "crashes": crashes,
            "goals": goals, "total_goals": goals.sum(), "arrived": arrived,
            "arrived_count": arrived.sum(), "done": done_after,
            "crashes_per_car": crashes.mean(), "ped_hits": ped_hit.sum(),
            "mean_speed": jnp.mean(speed)}
    return nst, _observe(env, nst, _candidates(env, pos)), reward, t >= env.max_steps, info


def make_env(pool: RoutePool, world_min, world_max, cell_size=35.0, cap=16, **kw) -> Env:
    import numpy as np
    seg = np.linalg.norm(np.diff(pool.xy, axis=1), axis=-1)          # (P, W-1)
    cum = np.concatenate([np.zeros((pool.xy.shape[0], 1)), np.cumsum(seg, axis=1)], 1)
    ncx, ncy = spatial.grid_dims(world_min, world_max, cell_size)
    return Env(
        routes_xy=jnp.asarray(pool.xy), routes_n=jnp.asarray(pool.n),
        routes_node=jnp.asarray(pool.node), routes_junc=jnp.asarray(pool.junc),
        routes_lanes=jnp.asarray(pool.lanes), routes_speed=jnp.asarray(pool.speed),
        routes_cum=jnp.asarray(cum, jnp.float32),
        world_min=jnp.asarray(world_min, jnp.float32),
        world_max=jnp.asarray(world_max, jnp.float32),
        cell_size=cell_size, cap=cap, ncx=ncx, ncy=ncy,
        cand_C=spatial.candidate_count(cap), **kw,
    )
