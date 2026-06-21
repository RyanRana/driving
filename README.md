# SmoothRide — RL traffic coordination for a fully-autonomous SF (2040)

**Thesis:** in a world where every car is autonomous, the human-era traffic system (signals, rigid lane law) becomes unnecessary — a coordinated swarm can negotiate right-of-way directly. We learn that **swarm-coordination / behavioral layer** with multi-agent RL on the **real San Francisco road graph**, the one irreducible human element being **pedestrians**, whom cars must slow for. Low-level autonomy (perception, control) is assumed solved; this project is the behavioral policy.

This README is the team entry point. Deeper detail: design specs in `docs/superpowers/specs/`, the overnight experiment log in `docs/HANDOFF-overnight.md`.

---

## Headline results (v2, 2026-06-21)

| Metric | Result | Model (checkpoint in Modal volume `smoothride-nav-ckpts`) |
|---|---|---|
| **In-distribution crashes** | **0.07% / car** (~1 per 1,400) | `trained_v5c96p5x` (96 cars / 5 peds, downtown) |
| **Held-out generalization (leave-one-out)** | **~1% / car** on unseen Mission | `trained_v4loo` (trained downtown+nopa+chinatown) |
| v1 baseline (for contrast) | 12% on Mission | `trained_peds` |

≈ **12× cross-map safety improvement** over v1. Both the ≤0.5% in-distribution target and cross-region generalization were achieved.

### The density frontier (key planning takeaway)
Near-zero crash requires **≤ ~96 cars AND ≤ ~5 pedestrians** in the downtown bbox. There are **two independent density walls**:
- **Car–car:** ~300 cars saturates the street graph → ~50% crash regardless of pedestrians. **300+ cars cannot be near-zero-crash** in this map.
- **Car–ped:** ~300 pedestrians (crossing at intersections) → high car–ped crashes.

So the honest safe operating point is **~80–100 cars + a handful of pedestrians**, not 300 cars. To scale density, you'd need a larger/denser road graph or relaxed geometry — that's a future direction, not a tuning fix.

---

## How it works (the RL setup)

- **Env** (`smoothride/env/kinematic.py`): vectorized JAX kinematic-bicycle multi-agent sim on the real SF graph. N cars follow routes; M pedestrians follow **deterministic hard-coded paths** (sidewalk run + a perpendicular crossing **at an intersection**), with only their start time randomized. Cars can brake to a full stop and resume.
- **Observation** (per car, ego-relative, permutation-invariant): ego state + **Deep Sets encoders** over the nearby cars and pedestrians (padded + masked sets). This is what makes the policy generalize across maps. (An attention/social-attention encoder is selectable via `--arch attention` but was **not** better than Deep Sets.)
- **Reward** (`§9`, efficiency only): progress + arrival − time. **All safety/comfort flows through a cost channel**, not the reward (CMDP).
- **Cost** (`smoothride/rl/verifier.py`, pure & deterministic — the "rules"): a **dual channel**:
  - **hard** = collisions (`car_crash` + `ped_hit`, pedestrian collisions weighted heavier) → Lagrangian target **0**.
  - **soft** = graded terms: **car-collision-risk hinge** (dense "back off" signal — the single biggest lever for eliminating car–car crashes), **pedestrian-yield hinge** (slow near crossers), lane-keeping, wrong-way, over-cap.
- **Trainer** (`smoothride/rl/ppo.py`): PPO with **dual-Lagrangian** (`reward − λ_hard·cost_hard − λ_soft·cost_soft`); λ's ascend toward their targets. `--regions a,b,c` round-robins regions per iteration for leave-one-out training.

**What made crashes drop:** the graded car-collision-risk hinge (turned a sparse binary crash signal into a dense gradient), the dual channel (crashes target 0 without flattening yielding), a low cruise cap (more reaction time), low density, and multi-region training (generalization).

---

## Reproduce / run

Prereqs: `python3` env with JAX/Flax/Optax + `modal` (authed). Training runs on Modal GPUs; eval/render run locally.

**Train (example — the in-distribution champion config):**
```bash
modal run --detach -m smoothride.rl.modal_train \
  --iters 400 --worlds 16 --agents 96 --n-peds 5 --steps 250 \
  --crash-target 0.0 --soft-target 0.05 --w-carped 8.0 --cruise-cap 4.0 \
  --region downtown --tag _demo --snapshot-every 100
```
Multi-region LOO: add `--regions downtown,nopa,chinatown_fidi` (instead of `--region`). Attention encoder: add `--arch attention`.

**Evaluate on a held-out region** (reports arrivals + per-step/any-step crash, off-lane, wrong-way):
```bash
modal volume get smoothride-nav-ckpts trained_v4loo.msgpack runs/trained_v4loo.msgpack
cp runs/trained_v4loo.msgpack runs/untrained_v4loo.msgpack   # eval compares trained vs untrained
python3 scripts/eval_policy.py --region mission --agents 96 --peds 10 --steps 250 \
  --trained runs/trained_v4loo.msgpack --untrained runs/untrained_v4loo.msgpack
```

**Render a viewer scene** (self-contained `scene.json` with terrain/buildings/trajectory) — see "Viewer" below.

Named regions live in `smoothride/data/map_loader.py::SF_REGIONS` (downtown, mission, nopa, chinatown_fidi). NOTE: a fixed-cache bug that made `--region` silently always load downtown was fixed (commit `99a9f9e`); the graph is now cached per-bbox.

---

## Viewer (Cesium 3D)

`smoothride/demo/cesium/` — static site. `public/scene.json` (or `public/manifest.json` for a multi-scene iteration **dropdown**) drives it.
```bash
cd smoothride/demo/cesium && python3 -m http.server 8137   # open http://127.0.0.1:8137
```
Needs an ion token in the git-ignored `config.js` (copy `config.example.js`). Cars: red=crashed / green=arrived / blue=en-route (speed-tinted); pedestrians: amber cylinders. Scenes are produced by `scripts/export_snapshots.py` (renders a series of checkpoints + a manifest) or `smoothride/demo/export_cesium.py` (single scene).

A **Mission demo of the generalization champion (`v4loo`)** has been rendered for the team (see `docs/HANDOFF-overnight.md` for the file / dropdown entry).

---

## Repo map

- `smoothride/env/` — env (`kinematic.py`), pedestrian paths (`ped_paths.py`), routing, spatial hash, map loader.
- `smoothride/rl/` — `verifier.py` (rules/cost), `ppo.py` (trainer), `networks.py` (Deep Sets + attention), `modal_train.py` (Modal entry + CLI), `trace.py`.
- `smoothride/demo/` — Cesium viewer + scene exporters.
- `scripts/` — `eval_policy.py` (held-out eval), `export_snapshots.py`, smoke tests.
- `tests/` — pytest suite (env, rl, data, demo); ~170+ tests.
- `docs/superpowers/specs/` — design specs (pedestrian-yield env, v2 crash-split). `docs/superpowers/plans/` — implementation plans.
- `docs/HANDOFF-overnight.md` — full overnight experiment log (every config + result; the authoritative record of the v2 campaign).

---

## What's next (for the team / demo)
- **Demo:** load the rendered Mission champion scene in the viewer; press play, zoom into an intersection to see cars slow for pedestrians and queue without colliding.
- Optional polish: `v2-T4` (end-on-all-done eval trim — kills the static tail after all cars finish) is specced but not built.
- To push held-out crash below ~1% or support higher density, the lever is environment/map scale, not more cost tuning (the frontier is density-bound).
