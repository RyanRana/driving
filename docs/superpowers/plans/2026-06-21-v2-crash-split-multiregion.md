# v2 — crash-channel split, intersection crossings, end-on-all-done, multi-region

**Date:** 2026-06-21 · **Branch:** `worktree-3d-sim-setup` · executes via subagent-driven-development.

**Goal:** drive crashes (esp. car–pedestrian) toward 0 as a hard CMDP constraint, make pedestrians cross at intersections, end episodes when all cars finish (eval/viewer), and train across multiple SF regions for safe cross-map generalization.

## Global constraints
- Immutability; verifier stays pure NumPy (no env import / RNG / wall-clock / network / LLM).
- JAX shape discipline (fixed shapes; pad+mask for variable counts). Determinism: ped motion pure fn of t.
- Reward stays §9 efficiency-only; ALL constraints flow through the cost channel(s).
- One rulebook: cost predicates live in `verifier`, reused by training relabel (`ppo.verifier_cost`) and offline grade (`verifier.cost_signal`).
- PEP 8, type annotations, functions < 50 lines. TDD per task.

## Design decisions
- **Two cost channels.** `cost_hard` = collisions: `w_carcar·car_crash + w_carped·ped_hit` (defaults `w_carcar=1.0`, `w_carped=3.0` — hitting a person is worse). `cost_soft` = `off_lane + wrong_way + over_cap + ped_yield` (graded). Two Lagrange multipliers `lam_hard`, `lam_soft`; `reward_eff = reward − lam_hard·cost_hard − lam_soft·cost_soft`. Dual ascent to two targets: `crash_target≈0.0`, `soft_target≈0.05`.
- **Expose `ped_hit` separately.** The env already computes `car_crash` and `ped_hit` separately in `step`; surface `ped_hit` in `info`/logging so the hard channel can weight car–ped collisions higher and metrics can report them.
- **Intersection crossings.** `build_ped_paths` chooses a road segment ending at (or nearest to) a junction node (`routes_junc`) and crosses there, instead of a random segment midpoint.
- **End-on-all-done.** Add an episode "all done" notion; trim eval/render trajectories at the last step any car is still en-route. Training keeps a fixed `max_steps` cap (scan needs static length; post-done steps already inert).
- **Multi-region round-robin.** `modal_train` builds an env per region in a `--regions` list and cycles region per iteration; shared policy params; verifier relabel uses that iter's env.

---

## Task 1 — Pedestrian crossings at intersections
**Files:** `smoothride/env/ped_paths.py` (modify `build_ped_paths`), `tests/env/test_ped_paths.py` (extend).
**Change:** `build_ped_paths` gains access to `routes_junc (P,W) bool` (pass it in alongside `routes_xy/n/lanes`). For each ped, pick a route+segment whose near endpoint is a **junction** (`routes_junc[r, w]` or `[r, w+1]` True); place the crossing leg at that junction point (perpendicular across the road there). Fall back to the old random-segment behavior if a route has no junction waypoints. Keep the polyline shape (4 pts), `cross_lo/cross_hi`, `starts`, determinism (seeded), and the `PedPaths` contract unchanged. Update `make_env`'s `build_ped_paths(...)` call to pass `pool.junc`.
**Tests:** with a fixture whose route has a junction waypoint, assert each ped's crossing leg midpoint is within a small tolerance of a junction node position; determinism holds; fallback works when no junctions.

## Task 2 — Verifier cost components + expose ped_hit
**Files:** `smoothride/rl/verifier.py`, `smoothride/env/kinematic.py` (expose `ped_hit` in `info`), `smoothride/rl/trace.py` (+ `ped_hit` optional field, default zeros for back-comx), `tests/rl/test_verifier_pedyield.py` / new test.
**Change:**
- `kinematic.step`: add `ped_hit` to the returned `info` (already computed as `ped_hit` before `crash_event`). Keep `just_crashed` (combined) as-is.
- `verifier.step_cost_components(...) -> dict` returning `{"car_crash","ped_hit","off_lane","wrong_way","over_cap","ped_yield"}` each `(T,N)` float. `car_crash` needs car-vs-ped split: accept separate `car_crashed` and `ped_hit` arrays (caller supplies). Keep existing `step_cost` (summed) working for `cost_signal` back-compat.
- Add `hard_cost(components, w_carcar=1.0, w_carped=3.0)` and `soft_cost(components)` helpers.
**Tests:** components sum matches the old `step_cost`; `hard_cost` weights ped_hit higher; soft excludes collisions.

## Task 3 — PPO dual-channel + modal_train dual-Lagrangian
**Files:** `smoothride/rl/ppo.py` (`collect` log `ped_hit`; `verifier_cost` → return/attach `cost_hard`,`cost_soft`; `update` take `lam_hard,lam_soft`), `smoothride/rl/modal_train.py` (two multipliers, two targets, dual ascent, logging), `tests/rl/test_ppo_smoke.py` (update to dual-channel).
**Change:**
- `collect` out-dict: add `ped_hit=info["ped_hit"]` (and keep `crashed`). Note `crashed` currently = combined; derive `car_crash = crashed & ~ped_hit`-equivalent OR log `car_crash` too. Simplest: log both `car_crash=info["just_crashed"] & ~ped_hit`… but just_crashed includes ped — cleaner: have env expose `car_crash` and `ped_hit` both in info. (Coordinate with Task 2's env change: expose BOTH `car_crash` and `ped_hit` in info.)
- `verifier_cost(env, batch)`: compute components, return `(cost_hard, cost_soft)` (or set `batch["cost_hard"]`, `batch["cost_soft"]`). Use `env`'s `ped_radius/r_yield/cruise_cap` and weights.
- `update(env, cfg, ts, batch, lam_hard=0.0, lam_soft=0.0)`: `reward_eff = batch["reward"] − lam_hard·batch["cost_hard"] − lam_soft·batch["cost_soft"]`.
- `modal_train`: track `lam_hard, lam_soft`; each iter `mean_hard, mean_soft = cost_hard.mean(), cost_soft.mean()`; dual ascent `lam_* = clip(lam_* + k·(mean_* − target_*), 0, 400)`; CLI `--crash-target` (default 0.0) and `--soft-target` (default 0.05); log both lams + both costs + a separate `ped_hit_rate` metric.
**Tests:** smoke runs `collect → verifier_cost → update(lam_hard, lam_soft)` and metrics finite; cost_hard/cost_soft shapes `(B,T,N)`.

## Task 4 — End-on-all-done trim (eval/render)
**Files:** `smoothride/rl/render.py` (or wherever the rollout dict is produced) + `scripts/export_snapshots.py` / `scripts/export_cesium.py` / `scripts/eval_policy.py` as needed; a small pure helper + test.
**Change:** add `last_active_step(arrived, crashed) -> int` (pure): the last step index where any car is neither arrived nor crashed (i.e., still en-route), +1; clamp ≥1. Trim the rollout/scene timeline to that length before building the scene / computing eval, so the run ends when the last car finishes. Training is untouched (fixed cap).
**Tests:** helper returns the right cut for hand-built arrived/crashed arrays (all-done early → short; never-done → full length).

## Task 5 — Multi-region round-robin training
**Files:** `smoothride/rl/modal_train.py`, (maybe a small region-cycling helper) + a unit test for the cycler.
**Change:** `--regions` (comma list, default just `--region`'s value for back-compat). Build a list of envs (one per region, each with its own route pool + ped paths). Each iteration: `env = envs[it % len(envs)]`; `collect`/`verifier_cost`/`update` use it. Snapshots/eval unaffected (single region for rendering). Log which region each iter used. Pure helper `region_for_iter(it, regions) -> str` unit-tested.
**Notes:** different regions → different array shapes, so we DON'T vmap across regions in one batch; we cycle whole envs across iterations (all 32 worlds share the iter's region). Policy params shared (ego-relative obs → transferable).

---

## After all tasks
- Full suite green.
- Retrain on Modal: dual-channel, multi-region (`--regions downtown,mission,nopa`), `--crash-target 0.0 --soft-target 0.05`, snapshots every 50, more iters. Then export + eval held-out, confirm crashes (esp. car–ped) ≈ 0 and cross-map crash gap closed.
