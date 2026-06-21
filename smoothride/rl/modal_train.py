"""Modal app: run the JAX nav-policy (MAPPO) training on a datacenter GPU.

Unlike the Isaac/PhysX low-level controller (see smoothride/lowlevel/modal_image.py),
the kinematic env + MAPPO learner are pure JAX — jit + vmap over many worlds on a
SINGLE device. So this needs only a light JAX image (no Isaac), and rollouts +
the PPO-Lagrangian update stay co-resident on one GPU (do NOT split rollout
workers from the learner — that's how the vmap design gets its throughput; see
docs/superpowers/specs/2026-06-20-sim-hosting.md).

Checkpoints persist to a modal.Volume so export_cesium / render pull them back.

One-time:
  pip install modal && modal token new

Run a scaled training (heavy density example):
  modal run -m smoothride.rl.modal_train --iters 400 --worlds 64 --agents 96 --n-peds 32

Pull the trained policy back for rendering:
  modal volume get smoothride-nav-ckpts trained.msgpack runs/trained.msgpack
  modal volume get smoothride-nav-ckpts untrained.msgpack runs/untrained.msgpack
  python -m smoothride.demo.export_cesium --elevation synthetic --agents 96 \\
      --out smoothride/demo/cesium/public/scene.json

Pull versioned snapshot checkpoints for scene-series export:
  modal volume get smoothride-nav-ckpts 'trained_it*.msgpack' runs/
  python scripts/export_snapshots.py --tag "" --elevation synthetic

The training loop here mirrors smoothride/rl/train_local.py::main, but writes to
the persistent volume and reports per-iter metrics in the Modal logs.
"""
from __future__ import annotations

import modal

APP_NAME = "smoothride-nav"
GPU = "A100"                  # H100 also fine; the env is light, A100 is plenty
TIMEOUT_S = 4 * 60 * 60
CKPT_DIR = "/ckpts"

app = modal.App(APP_NAME)

# Trained/untrained params persist here across runs (resume + export read from it).
volume = modal.Volume.from_name("smoothride-nav-ckpts", create_if_missing=True)

# Light JAX image — just the nav stack, no Isaac. Source is added at runtime so
# editing the trainer doesn't rebuild the image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]>=0.4.30", "flax>=0.8", "optax>=0.2",
        "osmnx>=2.0", "networkx>=3.0", "shapely>=2.0", "geopandas>=1.0",
        "pandas>=2.0", "numpy>=1.26", "pyproj>=3.6",
    )
    .add_local_python_source("smoothride")
)


def should_snapshot(it: int, snapshot_every: int, iters: int) -> bool:
    """Return True when iteration *it* should emit a versioned snapshot.

    Rules (evaluated in priority order):
      1. ``it == iters - 1`` (final iter) → always True, regardless of
         ``snapshot_every``.  The final policy is always captured.
      2. ``snapshot_every <= 0`` → mid-run snapshots disabled; only the final
         iter (rule 1) triggers.
      3. ``it == 0`` → the baseline (pre-training) policy; True when
         ``snapshot_every > 0``.
      4. ``it % snapshot_every == 0`` → periodic mid-run snapshot.

    ``snapshot_every > 0`` enables full versioning (iter 0 baseline +
    every N iters + final).  ``snapshot_every <= 0`` writes only the final
    iter, which costs one extra ``volume.commit()`` but keeps the volume tidy
    for quick experiments.
    """
    if it == iters - 1:
        return True
    if snapshot_every <= 0:
        return False
    return it == 0 or it % snapshot_every == 0


def snapshot_name(tag: str, it: int) -> str:
    """Return the versioned checkpoint filename for iteration *it*.

    Example: snapshot_name("_pedtest", 50) -> "trained_pedtest_it00050.msgpack"
    The five-digit zero-padded iter supports up to 99999 iters before overflow;
    beyond that the field expands naturally (no truncation).
    """
    return f"trained{tag}_it{it:05d}.msgpack"


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={CKPT_DIR: volume})
def train(iters: int = 300, worlds: int = 64, agents: int = 64,
          steps: int = 300, vmax: float = 16.0, routes: int = 1024,
          lagrangian: bool = True, crash_target: float = 0.0, seed: int = 0,
          verifier: bool = True, cost_target: float = 0.05, region: str = "downtown",
          tag: str = "", n_peds: int = 300, cruise_cap: float = 7.0,
          ped_radius: float = 3.5, cand_cap: int = 16,
          snapshot_every: int = 50, soft_target: float = 0.05,
          w_carped: float = 3.0, arch: str = "deepsets") -> dict:
    """Train the shared-weight nav policy; write {untrained,trained}{tag}.msgpack
    to the volume. Returns the final-iteration metrics dict.

    Dual-Lagrangian training (v2 Task 3):
      * ``lam_hard`` ascends toward ``crash_target`` (default 0.0 — zero crashes).
      * ``lam_soft`` ascends toward ``soft_target`` (default 0.05 — graded hinges).
    Both multipliers are clipped to [0, 400].

    ``cost_target`` is accepted for backward compat and maps to ``soft_target``.

    Additionally writes versioned snapshots ``trained{tag}_it{N:05d}.msgpack``
    at iter 0 (baseline), every ``snapshot_every`` iters, and the final iter.
    Set ``snapshot_every=0`` to disable mid-run snapshots (only final written).
    All existing periodic saves (every-10-iter ``trained{tag}.msgpack``) are
    preserved — versioned snapshots are purely additive.

    Set encoder selection (v2 Task 6):
      * ``arch="deepsets"`` (default) — masked mean+max pool (DeepSets).
      * ``arch="attention"`` — ego-query masked attention (AttentionPool).
        Example: ``modal run -m smoothride.rl.modal_train --arch attention``.
    """
    import json
    import os
    import time

    import jax
    from flax import serialization

    from smoothride.data.map_loader import SF_REGIONS, load_road_network
    from smoothride.env import kinematic as K
    from smoothride.env.routing import build_route_pool
    from smoothride.rl import ppo

    # --cost-target is the legacy alias for --soft-target; honour it if explicitly set.
    effective_soft_target = soft_target if soft_target != 0.05 else cost_target

    def save(ts, name: str) -> None:
        with open(os.path.join(CKPT_DIR, name), "wb") as f:
            f.write(serialization.to_bytes(ts.params))

    bbox = SF_REGIONS[region]                        # train region (named neighborhood)
    print(f"region={region} bbox={bbox}", flush=True)
    net = load_road_network(bbox=bbox)               # pulls + caches the SF graph
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=routes, seed=seed)
    # Lagrangian: zero the fixed crash penalty so the adaptive multiplier owns it.
    extra = {"w_collision": 0.0} if lagrangian else {}
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=agents, n_peds=n_peds,
                     max_steps=steps, v_max=vmax, cruise_cap=cruise_cap,
                     ped_radius=ped_radius, cand_cap_car=cand_cap,
                     cand_cap_ped=cand_cap, seed=seed, **extra)
    cfg = ppo.PPOConfig(n_worlds=worlds, encoder=arch)
    print(f"device={jax.devices()} env: agents={env.n_agents} obs={env.obs_dim} "
          f"steps={env.max_steps} worlds={cfg.n_worlds}", flush=True)
    print(f"dual-Lagrangian: crash_target={crash_target} soft_target={effective_soft_target} "
          f"w_carped={w_carped}", flush=True)

    key = jax.random.PRNGKey(seed)
    key, kinit = jax.random.split(key)
    ts = ppo.make_train_state(env, cfg, kinit)
    save(ts, f"untrained{tag}.msgpack")             # baseline shadow world
    volume.commit()

    # Dual-Lagrangian training (v2 Task 3): two multipliers ascend independently
    # toward their respective targets. lam_hard→crash_target (default 0 = zero crashes).
    # lam_soft→soft_target (graded hinges stay bounded). verifier=False disables
    # the verifier relabelling and falls back to the old crash-only path.
    history: list[dict] = []
    lam_hard: float = 0.0
    lam_soft: float = 0.0

    for it in range(iters):
        key, kc = jax.random.split(key)
        t0 = time.time()
        batch = ppo.collect(env, ts, kc, cfg.n_worlds)

        if verifier:
            cost_hard, cost_soft = ppo.verifier_cost(
                env, batch, w_carped=w_carped,
            )                                        # each (B,T,N), off-device
            batch = {**batch, "cost_hard": cost_hard, "cost_soft": cost_soft}
            mean_hard = float(cost_hard.mean())
            mean_soft = float(cost_soft.mean())

            if lagrangian:
                # Dual ascent: hard channel (collisions → 0) is aggressive (step 5.0)
                # because we want zero crashes; soft channel uses a smaller step (2.0).
                lam_hard = float(
                    min(400.0, max(0.0, lam_hard + 5.0 * (mean_hard - crash_target)))
                )
                lam_soft = float(
                    min(400.0, max(0.0, lam_soft + 2.0 * (mean_soft - effective_soft_target)))
                )

        if verifier:
            ts, m = ppo.update(env, cfg, ts, batch,
                               lam_hard=lam_hard if lagrangian else 0.0,
                               lam_soft=lam_soft if lagrangian else 0.0)
        else:
            ts, m = ppo.update(env, cfg, ts, batch)

        m = {k: float(v) for k, v in m.items()}
        m["iter"] = it
        m["sec"] = round(time.time() - t0, 2)

        if verifier:
            import numpy as _np
            m["cost_hard"] = round(mean_hard, 4)
            m["cost_soft"] = round(mean_soft, 4)
            # Per-collision-type rates from the raw batch flags (before weighting).
            m["car_ped_rate"] = round(
                float(_np.asarray(batch["ped_hit"]).mean()), 4
            )
            m["car_car_rate"] = round(
                float(_np.asarray(batch["car_crash"]).mean()), 4
            )

        if lagrangian:
            m["lam_hard"] = round(lam_hard, 2)
            m["lam_soft"] = round(lam_soft, 2)

        history.append(m)

        if it % 10 == 0 or it == iters - 1:
            lam_s = (f"lam_h {lam_hard:6.2f} lam_s {lam_soft:6.2f} | "
                     if lagrangian else "")
            cost_s = (f"hard {m.get('cost_hard', 0):.3f} "
                      f"soft {m.get('cost_soft', 0):.3f} | "
                      if verifier else "")
            rate_s = (f"car-car {m.get('car_car_rate', 0):.3f} "
                      f"car-ped {m.get('car_ped_rate', 0):.3f} | "
                      if verifier else "")
            print(f"it {it:4d} | reward {m['ep_reward']:8.1f} | {lam_s}{cost_s}"
                  f"{rate_s}crashes/car {m['crashes_per_car']:.2f} | "
                  f"goals/agent {m['goals_per_agent']:.2f} | {m['sec']}s", flush=True)
            save(ts, f"trained{tag}.msgpack")        # periodic, so renders mid-run
            with open(os.path.join(CKPT_DIR, f"history{tag}.json"), "w") as f:
                json.dump(history, f)
            volume.commit()

        # Versioned snapshot — additive on top of the existing periodic saves above.
        if should_snapshot(it, snapshot_every, iters):
            name = snapshot_name(tag, it)
            save(ts, name)
            volume.commit()
            print(f"  snapshot -> {name}", flush=True)

    save(ts, f"trained{tag}.msgpack")
    with open(os.path.join(CKPT_DIR, f"history{tag}.json"), "w") as f:
        json.dump(history, f)
    volume.commit()
    print(f"done: untrained{tag}.msgpack, trained{tag}.msgpack -> volume "
          f"{APP_NAME}-ckpts", flush=True)
    return history[-1]


@app.local_entrypoint()
def main(iters: int = 300, worlds: int = 64, agents: int = 64,
         steps: int = 300, lagrangian: bool = True, verifier: bool = True,
         cost_target: float = 0.05, region: str = "downtown", tag: str = "",
         wait: bool = False, n_peds: int = 300, cruise_cap: float = 7.0,
         ped_radius: float = 3.5, cand_cap: int = 16, seed: int = 0,
         snapshot_every: int = 50, crash_target: float = 0.0,
         soft_target: float = 0.05, w_carped: float = 3.0,
         arch: str = "deepsets"):
    """Local entrypoint: spawns (or waits on) the remote training function.

    New flags (v2 Task 3 — dual-Lagrangian):
      --crash-target  Hard collision target (default 0.0; drive to zero crashes).
      --soft-target   Graded hinge target (default 0.05; mirrors old --cost-target).
      --w-carped      Car-ped weight in hard_cost (default 3.0; car-ped > car-car).

    New flag (v2 Task 6 — selectable encoder):
      --arch          Set encoder: "deepsets" (default) or "attention".
                      Example: modal run -m smoothride.rl.modal_train --arch attention

    Legacy flag (still accepted):
      --cost-target   Maps to --soft-target for backward compat.
    """
    kw = dict(iters=iters, worlds=worlds, agents=agents, steps=steps,
              lagrangian=lagrangian, verifier=verifier, cost_target=cost_target,
              region=region, tag=tag, n_peds=n_peds, cruise_cap=cruise_cap,
              ped_radius=ped_radius, cand_cap=cand_cap, seed=seed,
              snapshot_every=snapshot_every, crash_target=crash_target,
              soft_target=soft_target, w_carped=w_carped, arch=arch)
    if wait:                       # blocking: streams live, dies if the client drops
        print("final metrics:", train.remote(**kw))
        return
    # Default: SPAWN server-side so a flaky local connection can't cancel the run
    # (modal warns .remote() in detached apps is canceled on client disconnect).
    # Launch with `modal run --detach` so the app outlives this client. Checkpoints
    # (saved every 10 iters) land in the volume regardless.
    fc = train.spawn(**kw)
    print(f"spawned training (call {fc.object_id}); region={region} tag={tag}\n"
          f"  dual-Lagrangian: crash_target={crash_target} soft_target={soft_target} "
          f"w_carped={w_carped}\n"
          f"  checkpoints -> volume '{APP_NAME}-ckpts' (untrained{tag}.msgpack / trained{tag}.msgpack)\n"
          f"  versioned snapshots every {snapshot_every} iters -> trained{tag}_it####.msgpack\n"
          f"  pull when done:  modal volume get {APP_NAME}-ckpts trained{tag}.msgpack runs/trained{tag}.msgpack")
