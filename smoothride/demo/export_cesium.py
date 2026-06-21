"""Export a kinematic rollout to a Cesium scene JSON (schema v1).

Reuses the reprojection from export_web and the rollout from render; adds a baked
ground `z` per car/ped by nearest-node elevation lookup, and writes via scene.py.

Usage:
  python -m smoothride.demo.export_cesium \
      --trained runs/trained.msgpack --untrained runs/untrained.msgpack \
      --agents 24 --peds 12 --steps 300 \
      --out smoothride/demo/cesium/public/scene.json
"""
from __future__ import annotations

import argparse
import os

import jax
import numpy as np

from ..data.map_loader import attach_elevation, load_road_network
from ..env import kinematic as K
from ..env.routing import build_route_pool
from . import scene as S
from .export_web import _lonlat_transformer, _roads_geojson, _to_lonlat
from .render import load_params, rollout

CESIUM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cesium"))
DEFAULT_OUT = os.path.join(CESIUM_DIR, "public", "scene.json")


def sample_path_z(pos: np.ndarray, node_xy: np.ndarray, node_z: np.ndarray) -> np.ndarray:
    """Nearest-node elevation for every point in pos (T, N, 2) -> (T, N) z."""
    T, N, _ = pos.shape
    flat = pos.reshape(-1, 2)
    d2 = ((flat[:, None, :] - node_xy[None, :, :]) ** 2).sum(-1)   # (T*N, nodes)
    nearest = d2.argmin(1)
    return node_z[nearest].reshape(T, N)


def _roads_3d(net, tf):
    """2D road segments + a baked z per endpoint (nearest-node elevation)."""
    flat2d = _roads_geojson(net, tf)                      # [[[lon,lat],[lon,lat]], ...]
    segs_xy = net.node_xy[net.edges]                      # (E, 2, 2) meters
    z = net.node_z[net.edges]                             # (E, 2)
    out = []
    for e, seg in enumerate(flat2d):
        out.append([[seg[0][0], seg[0][1], round(float(z[e, 0]), 2)],
                    [seg[1][0], seg[1][1], round(float(z[e, 1]), 2)]])
    return out


def build_from_rollouts(net, env, tf, rollouts: dict, stride: int) -> dict:
    worlds = {}
    for name, tr in rollouts.items():
        car_lon, car_lat = _to_lonlat(net, tf, tr["pos"])
        car_z = sample_path_z(tr["pos"], net.node_xy, net.node_z)
        if tr["ped"].shape[1] > 0:
            ped_lon, ped_lat = _to_lonlat(net, tf, tr["ped"])
            ped_z = sample_path_z(tr["ped"], net.node_xy, net.node_z)
        else:
            ped_lon = ped_lat = ped_z = np.zeros((tr["pos"].shape[0], 0))
        worlds[name] = S.pack_world(
            car_lon=car_lon, car_lat=car_lat, car_z=car_z,
            heading=tr["heading"], speed=tr["speed"],
            crashed=tr["crashed"], goals=tr["goals"], arrived=tr.get("arrived"),
            ped_lon=ped_lon, ped_lat=ped_lat, ped_z=ped_z, stride=stride)
    return worlds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", default="runs/trained.msgpack")
    ap.add_argument("--untrained", default="runs/untrained.msgpack")
    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--peds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--elevation", default="3dep", choices=["3dep", "synthetic"])
    ap.add_argument("--buildings", action="store_true", help="pull OSM buildings")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    net = attach_elevation(load_road_network(), source=args.elevation)
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=1024, seed=0)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                     n_peds=args.peds, max_steps=args.steps)
    tf = _lonlat_transformer(net)

    rollouts = {}
    for name, ckpt in [("trained", args.trained), ("untrained", args.untrained)]:
        params = load_params(env, ckpt)
        rollouts[name] = rollout(env, params, jax.random.PRNGKey(args.seed), sample=True)

    worlds = build_from_rollouts(net, env, tf, rollouts, args.stride)

    corners = np.array([[x0, y0], [x1, y1]], np.float32)
    clon, clat = _to_lonlat(net, tf, corners)
    meta = {
        "dt": float(env.dt) * args.stride,
        "n_steps": len(range(0, args.steps, args.stride)),
        "vmax": float(env.v_max),
        "center": [round(float(clon.mean()), 6), round(float(clat.mean()), 6)],
        "bounds": [[round(float(clon[0]), 6), round(float(clat[0]), 6)],
                   [round(float(clon[1]), 6), round(float(clat[1]), 6)]],
        "zoom": 15.5,
    }
    buildings = {"type": "FeatureCollection", "features": []}
    if args.buildings:
        from ..data.buildings import fetch_buildings_geojson
        buildings = fetch_buildings_geojson((-122.4180, 37.7820, -122.4000, 37.7950))

    scene = S.build_scene(meta=meta, roads=_roads_3d(net, tf),
                          buildings=buildings, worlds=worlds)
    nbytes = S.write_scene(args.out, scene)
    print(f"saved {args.out} ({nbytes/1024:.0f} KB, {len(scene['roads'])} road segs, "
          f"{len(buildings['features'])} buildings)")


if __name__ == "__main__":
    main()
