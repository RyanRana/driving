# SmoothRide — Where to Host the RL Loop (Hosting Research + Recommendation)

**Date:** 2026-06-20
**Status:** Draft for review
**Companion to:** `docs/HANDOFF-sim-contract.md` (§1, §8), `docs/superpowers/specs/2026-06-20-rl-env-reframe-design.md` (§8, §11, §12)

---

## TL;DR

Host the **experiment → verify → reward → update** loop on **a single GPU**, not on a fan-out cluster — because the actual learner (`rl/ppo.py`) is one tightly-coupled JAX process (`collect` → `update`) on one device, not a network of rollout workers around a separate learner. **Primary recommendation: a phased path** — (1) **local / the user's Mac** for dev-loop smoke tests (tiny `--iters`, the env is pure-JAX and macOS-runnable), (2) **Modal single-GPU function** for scaled training runs, reusing the Modal plumbing the repo already has, with `modal.Volume` as the msgpack checkpoint store. The **verifier and eval** are pure-CPU functions over a logged trace and should run wherever the trace lands (CI/CPU/Modal-CPU). **HUD is the wrong host for the trainer** (it is an LLM/computer-use-agent eval platform, not a vectorized-physics RL substrate) and should be treated only as a *candidate eval/verifier-publishing harness, pending confirmation* — not wired now.

---

## Workload characterization (grounded in the code)

What the loop in HANDOFF §8 / design §8 actually is, read off the source:

- **The env is pure JAX, GPU-friendly, one process.** `smoothride/env/kinematic.py` is a `reset`/`step` pytree env; `make_env`/`step` are `jit`-able and `vmap`-able over worlds (leaves `(B, N, ...)`). Rollouts are cheap and massively parallel **on a single device** — there is no inter-process rollout protocol.
- **The learner is single-GPU and tightly coupled.** `rl/ppo.py::collect` does `jax.vmap(one_world_rollout)` over `n_worlds` (default 32), then `rl/ppo.py::update` runs PPO epochs on the *same* batch on the *same* device. `train_local.py`'s loop is literally `for it in range(iters): batch = collect(...); ts, m = update(..., lam)`. **Collect and update share the device and the train state** — this is NOT a "many rollout workers + one learner" topology; splitting them across machines would add serialization cost for no throughput gain, because rollouts are already GPU-parallel via `vmap`.
- **PPO-Lagrangian, many iterations.** `update` subtracts `lam * cost` (`reward_eff = batch["reward"] - lam * batch["cost"]`); `train_local.py` does dual ascent on `lam` toward a `crash_target`. Each iteration is GPU-bound; you want many iterations fast, which means **low per-iteration latency on one fast GPU**, not horizontal scale-out.
- **Checkpoints are flax msgpack.** `train_local.py::save_params` writes `serialization.to_bytes(ts.params)`; `demo/render.py::load_params` reads `serialization.from_bytes(...)`. They are small param blobs (`untrained*.msgpack`, `trained*.msgpack`). The hosting must provide **durable shared storage** for these between a training run and the renderer/verifier — a volume or object store, nothing fancier.
- **The verifier is pure, CPU, no GPU, no LLM.** HANDOFF §7/§8 and design §10: `verify(trace) -> RunVerdict` is geometric predicates over logged arrays. "Same trace → same verdict." It can run *anywhere* — CI, a laptop, a Modal CPU function — and must NOT import the env or call Cosmos.
- **Two distinct GPU workloads already exist.** The repo already runs the **Isaac/PhysX low-level controller** on Modal (`lowlevel/modal_image.py`: `GPU = "A100"`, `modal.Volume "smoothride-lowlevel-ckpts"`, headless PhysX, launched with `modal run ... train_isaac.py`). The **kinematic JAX nav env** is the *light* workload — macOS-local-runnable, and the one this doc is sizing.

**Net:** the loop wants *one fast GPU with low dev-loop latency + a durable checkpoint volume + a CPU lane for the verifier*. It does **not** want a serverless fan-out cluster for the learner itself. (Fan-out is real, but for **offline Cosmos scenario generation / eval cohorts** — design §12 "offline Cosmos fan-out" — not for the PPO inner loop.)

---

## Options compared

| Option | Throughput fit (cheap-parallel JAX rollouts + 1 learner) | Cost (serverless vs always-on; idle) | Integration effort | Iteration speed / dev-loop latency | Loop mapping (HANDOFF §1/§8) |
|---|---|---|---|---|---|
| **Local / user's Mac** | Good for *tiny* runs; `vmap` over a handful of worlds on CPU/Metal. Caps out fast — no datacenter GPU. | Free; zero idle cost. | **Zero** — `python -m smoothride.rl.train_local --iters N`. Already works. | **Best** — no cold start, no sync. Instant edit→run. | Rollout+learner+verifier all in-process on the laptop. Checkpoints to local `runs/`. |
| **Modal (single-GPU function)** | Strong — one A100/H100 runs `collect`+`update` exactly as a `@gpu_function`. JAX rollouts stay device-parallel. | Per-second GPU billing (~$2.5/hr A100, ~$3.95/hr H100), **scale-to-zero = no idle cost**. Pay only while iterating. | **Low** — repo already has `modal_image.py`/`train_isaac.py` patterns + `modal.Volume`. Need a JAX image (lighter than the Isaac one) + a thin entrypoint. | Good, with a caveat: **cold start** (container spin-up, a few s) per launch; keep a session warm for tight loops, or batch iters per call. | Rollout+learner on the GPU function; checkpoints to `modal.Volume`; verifier can run as a sibling **CPU** Modal function or in CI off the committed trace. |
| **Rented persistent GPU box (Brev CLI)** | Strong — a single persistent A100/H100; same single-process loop, no cold start. Repo has a `brev-cli` skill (`brev create`, `brev shell`, `brev exec`). | **Always-on = idle cost** unless you stop it. Cheaper than Modal *if* utilization is high and continuous; wasteful for bursty dev. | Medium — provision box, sync code, manage env/CUDA, durable storage is your problem (scp/volume). | Very good *once up* — no per-call cold start; SSH/VS Code remote. Friction is provisioning + teardown discipline. | Rollout+learner on the box; checkpoints to box disk / synced volume; verifier on-box or in CI. |
| **HUD** | **Poor fit for the trainer.** HUD is an OSS RL-environment + **eval** platform for **computer-use / LLM agents** (browse web, edit spreadsheets, run terminal), built on an MCP "two-yield scenario" pattern (prompt → score → trajectory+reward for GRPO/RFT). It does **not** host a custom pure-JAX vectorized physics env + custom PPO-Lagrangian learner. | N/A for this workload. | High and mismatched — would mean reshaping our CMDP into an LLM-agent eval harness. | N/A. | At most: a **publishing/eval harness for the verifier side** (trace → verdict as a scored scenario), pending confirmation. Not the learner host. |

*Alternatives considered and dropped:* **RunPod / Lambda / vast.ai** — equivalent to "rented GPU box" but **without** the repo's existing `brev-cli` plumbing, so Brev represents that class with lower integration tax. **SkyPilot / Ray / Anyscale / managed k8s** — these are *distributed* schedulers; our learner is single-process single-GPU, so a cluster orchestrator is overkill and adds Ray-actor serialization the `vmap` design specifically avoids. They'd only earn their keep for the **offline Cosmos/eval fan-out**, which Modal already covers per design §12. (See [SkyPilot/Ray notes below](#open-questions).)

### Per-option notes

**Local / Mac.** The env is explicitly "light and macOS-local-runnable" (HANDOFF, task brief). For the *dev loop* — change reward shaping, re-run a 5–10 iteration smoke, eyeball `crashes/car` and `goals/agent` from `train_local.py`'s prints — nothing beats zero-latency local. It is the **inner dev loop**, not the scaled training run.

**Modal (single-GPU).** The repo already commits to Modal for GPU compute (design §12: "GPU compute — training rollouts + offline Cosmos fan-out"; `lowlevel/modal_image.py`). Per-second billing + scale-to-zero exactly matches *bursty* training: you pay for the hour you train and nothing overnight. The only real friction is **cold start** and **code-sync** per launch; mitigate by (a) a JAX-only image (no Isaac heaviness — much faster to build/pull than the `nvcr.io/nvidia/isaac-sim` image), (b) running many PPO iterations per remote call (the loop already takes `--iters`), and (c) `add_local_python_source("smoothride")` so editing the trainer doesn't rebuild the image (mirrors `modal_image.py`).

**Brev persistent box.** Best when you want a **long, continuous** training campaign and a stable remote dev environment (SSH/VS Code), and you'll keep the GPU busy. The trade is **idle cost discipline** — an always-on A100 left running overnight burns money Modal wouldn't. Good as a "campaign" tier *if* Modal cold-start friction proves annoying for multi-hour runs; otherwise Modal's scale-to-zero wins on cost for bursty work.

**HUD — honest assessment.** HUD ([hud.ai](https://www.hud.ai/), [hud-evals/hud-python](https://github.com/hud-evals/hud-python), YC W25) is RL **infrastructure for AI/computer-use agents**: define agent-callable tools, write eval scenarios, run evals at scale, feed successful trajectories into GRPO/RFT. Its unit of work is an **LLM agent acting in a software environment**, scored by a rubric. Our workload is a **vectorized JAX physics env with a custom PPO-Lagrangian dual-ascent learner** and a **deterministic geometric verifier** — none of which maps onto HUD's agent-eval/MCP model. **HUD is a poor host for the trainer.** Where it *could* conceivably fit is the **eval/verifier-publishing** side (our `verify(trace)` is already a deterministic scorer that yields a reward — superficially HUD's "scenario → score → trajectory+reward" shape). Treat that as **unconfirmed** and do not wire it before validating it can wrap a non-LLM, trace-in/verdict-out scorer.

---

## Recommendation + rationale

**Primary: a two-tier path that matches the loop's real shape (one fast GPU + a CPU verifier lane), reusing existing plumbing.**

1. **Dev loop → local / Mac.** Smoke-test reward/cost changes with `train_local.py --iters` on a few worlds. Zero latency, zero cost, zero new infra. This is where you iterate on the CMDP shaping and confirm `crashes/car` moves.
2. **Scaled training → Modal single-GPU function.** Promote the same loop to a `@gpu_function`-style entrypoint on A100/H100 with a JAX image and a `modal.Volume` for the msgpack checkpoints. Per-second + scale-to-zero billing fits bursty training; the repo already speaks Modal, so integration tax is low. **Do not split rollout-workers from the learner** — keep `collect`+`update` co-resident on one device, because that is how the JAX code is written and where its throughput comes from.
3. **Verifier / eval → CPU, wherever the trace is.** `verify(trace)` runs in CI or as a Modal **CPU** function off the committed/stored trace. It never needs the GPU and must stay decoupled from the env (HANDOFF §8).

**Optional campaign tier → Brev persistent GPU** *if* a multi-hour continuous run makes Modal's per-call cold start annoying and you can keep the box busy (and disciplined about stopping it). This is a swap-in for tier 2, not an addition.

**Rationale in one line:** the learner is single-process single-GPU by construction (`rl/ppo.py`), so the right host is "one fast GPU, billed only while training, with a durable param volume and a CPU verifier lane" — which is local→Modal, exactly the tools already in the repo. No cluster, no HUD-for-training.

---

## Mapping to the HANDOFF §1/§8 loop (which component runs where)

| Loop box (HANDOFF §8 / design §8) | Runs where | Backed by |
|---|---|---|
| **EXPERIMENT** — roll out current policy on a seeded world cohort | GPU: `rl/ppo.py::collect` (`vmap` over worlds) | Local for smoke; **Modal single-GPU** at scale |
| **RESULT** — the TRACE (per-step pose + events) | Written alongside the rollout (the rollout wrapper fills `Trace`, HANDOFF §7) | Stored in the checkpoint volume / object store next to params |
| **VERIFIER** — deterministic pass over the trace | **CPU**, decoupled | CI / Modal **CPU** function — never the GPU, never HUD-for-training |
| **REWARD** — −travel_time s.t. costs ≤ 0 (CMDP) | Derived from the verdict + cost channel | Pure function output; feeds the update |
| **FINE-TUNE** — MAPPO update + Lagrangian λ | GPU: `rl/ppo.py::update` + dual ascent in `train_local.py` | **Same device as EXPERIMENT** (co-resident) |
| **Checkpoint store** (the seam between iterations + the renderer) | Durable shared storage | `modal.Volume` (mirrors `smoothride-lowlevel-ckpts`) holding `*.msgpack`; `render.py::load_params` reads it |

Keeping EXPERIMENT and FINE-TUNE **on the same GPU** is the load-bearing decision — it is how the JAX code achieves throughput (HANDOFF §1: "rollouts are cheap and massively parallel"), and it is why a rollout-worker/learner split would be a regression here.

---

## Mapping to the design §12 tool-role table

Design §12 gives Modal one job: **"GPU compute — training rollouts + offline Cosmos fan-out."** This recommendation **keeps that role, sharpened**, and **carves out** the verifier per the spec's "every tool earns one artifact / one job each" principle:

- **Modal — keep + sharpen:** still "GPU compute," now explicitly **single-GPU co-resident rollout+learner for the nav policy** (not a fan-out cluster for the learner). Its offline-Cosmos fan-out role is unchanged and is where horizontal scale actually belongs. Modal's one artifact: **trained `*.msgpack` params in `modal.Volume`.**
- **Verifier — carve out to CPU (new row, implicit):** the deterministic verifier is *not* a GPU tool and should not ride on Modal's GPU role; it runs CPU-side (CI or Modal-CPU). Its one artifact: **the `RunVerdict` (reward + validity) over a trace.** Keeping it off the GPU host preserves the §10 "verify the trace, don't re-simulate" / hardware-independence guarantee.
- **HUD — do NOT add to §12 yet.** Adding HUD as a *training* tool would violate "every tool earns one non-overlapping job," since Modal already owns GPU compute and the deterministic verifier already owns reward/validity. HUD only earns a §12 row **if** it is confirmed as the *publishing/eval harness* for the verifier-as-scenario — a distinct artifact (a hosted, shareable eval). Until confirmed, it earns no row.
- **Brev — not a §12 tool;** it's an *infrastructure substrate* for Modal's same "GPU compute" job, swappable, not a new artifact.

This respects the §2 / §12 invariant: **one tool, one artifact.** Modal = trained params; verifier (CPU) = verdict; renderer (Cesium, §11) = the offline visual — no overlap.

---

## Open questions / confirm before wiring

1. **HUD's actual capability for non-LLM scorers.** Can HUD wrap a deterministic, trace-in/verdict-out function (no LLM agent, no MCP tool-calling) as a scored scenario? The docs frame everything around computer-use agents and the two-yield MCP pattern. **If not, HUD is out entirely** and the verifier stays plain CPU/CI. *I could not verify HUD supports a pure-function, non-agent scorer — treat as the eval/verifier candidate pending confirmation only.*
2. **Trace storage location.** Traces (HANDOFF §7) can be large per run. Confirm whether they live in the **same `modal.Volume`** as params, a separate object store (S3/GCS), or are committed to git for CI-side verification. The verifier host follows the trace.
3. **Modal cold-start tolerance.** Is per-launch cold start (a few seconds, plus image pull on first use) acceptable for the training cadence, or do multi-hour campaigns argue for a **Brev persistent box**? Decide the tier-2 vs campaign-tier swap.
4. **JAX-on-Modal image.** The existing `modal_image.py` is Isaac/PhysX-heavy. The nav trainer needs a **separate, lighter JAX image** (JAX + flax + optax + the `smoothride` source via `add_local_python_source`). Confirm GPU type (A100 vs H100) for the JAX learner — likely A100 is plenty given the env is light.
5. **Does the loop ever need true fan-out for the *learner*?** Only if you move to population-based training / many seeds in parallel. Today it doesn't (single `train_state`). If it does later, **many independent single-GPU Modal jobs** (one per seed) beats a Ray/SkyPilot cluster, staying consistent with the single-process design.

---

**Sources:** [HUD (hud.ai)](https://www.hud.ai/), [hud-evals/hud-python](https://github.com/hud-evals/hud-python), [HUD docs](https://docs.hud.ai/), [HUD on YC](https://www.ycombinator.com/companies/hud), [Modal pricing 2026 (CostBench)](https://costbench.com/software/ai-gpu-cloud/modal/), [Scale-to-zero serverless GPU: Modal vs RunPod vs Replicate](https://www.buildmvpfast.com/blog/scale-to-zero-serverless-gpu-modal-runpod-ai-hosting-2026). Repo files read: `docs/HANDOFF-sim-contract.md` §1/§7/§8, `docs/superpowers/specs/2026-06-20-rl-env-reframe-design.md` §8/§11/§12, `smoothride/lowlevel/modal_image.py`, `smoothride/lowlevel/train_isaac.py`, `smoothride/rl/ppo.py`, `smoothride/rl/train_local.py`, `smoothride/env/kinematic.py`, `smoothride/demo/render.py` (load_params), `~/.claude/skills/brev-cli/SKILL.md`.
