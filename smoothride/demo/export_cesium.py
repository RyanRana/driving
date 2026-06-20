"""Export a 3D rollout for the CesiumJS viewer (smoothride/demo/cesium/).

Mirrors export_web.py but for the 3D world: build the 3D ground truth (terrain +
buildings), roll out a policy on it, write a reproducible trace, run the
deterministic verifier, then pack per-car (lon, lat, height, heading, speed,
crash) trajectories + the verifier summary into trajectories.json.

Cesium streams the SF terrain (Cesium World Terrain) and 3D buildings (Cesium OSM
Buildings) from ion, so NO geometry is baked into the JSON — only the moving cars.

Usage:
  python -m smoothride.demo.export_cesium                     # random policy
  python -m smoothride.demo.export_cesium --ckpt runs/trained.msgpack
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import jax
import numpy as np

from ..data.map_loader import load_road_network_3d
from ..env import kinematic as K
from ..env import trace as TR
from ..env.routing import build_route_pool
from ..eval.verifier import verify
from .export_web import _lonlat_transformer, _to_lonlat

CESIUM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cesium"))
DEFAULT_OUT = os.path.join(CESIUM_DIR, "public", "trajectories.json")
TRACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                         "runs", "traces"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint; omit to use the scripted waypoint-follower")
    ap.add_argument("--policy", choices=["heuristic", "random"], default="heuristic",
                    help="fallback driver when no checkpoint is given")
    ap.add_argument("--agents", type=int, default=40)
    ap.add_argument("--peds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-buildings", action="store_true")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    net, buildings = load_road_network_3d(with_buildings=not args.no_buildings)
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=1024, seed=0)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                     n_peds=args.peds, max_steps=args.steps, buildings=buildings)
    print(f"world: {net.n_nodes} nodes, z {net.node_z.min():.0f}-{net.node_z.max():.0f} m, "
          f"{0 if buildings is None else buildings.n_buildings} buildings")

    # policy: real checkpoint if given; else a scripted controller (default) so
    # the cars actually drive the streets, or a random policy for stress-testing.
    run_id = f"cesium_{args.seed}"
    if args.ckpt and os.path.exists(args.ckpt):
        from .render import load_params
        params = load_params(env, args.ckpt)
        ckpt_id = os.path.basename(args.ckpt)
        roll = TR.rollout(env, params, jax.random.PRNGKey(args.seed), sample=True)
    elif args.policy == "random":
        ckpt_id = "random"
        params = TR.random_params(env, seed=args.seed)
        roll = TR.rollout(env, params, jax.random.PRNGKey(args.seed), sample=True)
    else:
        if args.ckpt:
            print(f"checkpoint {args.ckpt} not found — using the scripted driver")
        ckpt_id = "heuristic"
        roll = TR.rollout_heuristic(env, jax.random.PRNGKey(args.seed))

    manifest = TR.Manifest.from_env(env, run_id, args.seed,
                                    policy_checkpoint_id=ckpt_id)
    trace = TR.build_trace(env, manifest, roll)
    trace_path = TR.write_jsonl(trace, os.path.join(TRACE_DIR, f"{run_id}.jsonl"))
    verdict = verify(trace)
    print(f"trace -> {trace_path}")
    print(f"verifier: valid_run={verdict.valid_run} trips={verdict.trips} "
          f"crashes={verdict.crash_count} offroad={verdict.offroad_count} "
          f"rule={verdict.rule_count}")

    # reproject (x, y) -> (lon, lat); carry draped height z
    tf = _lonlat_transformer(net)
    pos = np.stack([roll["x"], roll["y"]], axis=-1)          # (T, N, 2)
    lon, lat = _to_lonlat(net, tf, pos)                      # (T, N)
    z, hdg, spd = roll["z"], roll["heading"], roll["speed"]
    crashed = np.cumsum(roll["crash"].astype(np.int32), axis=0) > 0

    T, N = lon.shape
    frames = range(0, T, args.stride)
    cars = []
    for i in range(N):
        cars.append({
            "lng": [round(float(lon[t, i]), 6) for t in frames],
            "lat": [round(float(lat[t, i]), 6) for t in frames],
            "h": [round(float(z[t, i]), 1) for t in frames],
            "hdg": [round(float(hdg[t, i]), 4) for t in frames],
            "spd": [round(float(spd[t, i]), 2) for t in frames],
            "crash": [int(crashed[t, i]) for t in frames],
        })

    corners = np.array([[x0, y0], [x1, y1]], np.float32)
    clon, clat = _to_lonlat(net, tf, corners)
    data = {
        "meta": {
            "dt": float(env.dt) * args.stride,
            "n_steps": len(list(frames)),
            "vmax": float(env.v_max),
            "center": [round(float(clon.mean()), 6), round(float(clat.mean()), 6)],
            "bounds": [[round(float(clon[0]), 6), round(float(clat[0]), 6)],
                       [round(float(clon[1]), 6), round(float(clat[1]), 6)]],
        },
        "summary": verdict.summary(),
        "manifest": dataclasses.asdict(manifest),
        "cars": cars,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    kb = os.path.getsize(args.out) / 1024
    print(f"saved: {args.out}  ({kb:.0f} KB, {N} cars x {len(list(frames))} frames)")


if __name__ == "__main__":
    main()
