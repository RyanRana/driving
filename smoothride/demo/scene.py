"""Scene schema v1 — the render/IO contract.

ONE file format produced by every sim backend (kinematic now, Isaac/PhysX later)
and consumed by the Cesium viewer. Trajectories are reprojected to lon/lat with a
baked ground `z`, rounded, and packed per-car to stay small.

A scene is:
  schema_version: int
  meta:   {dt, n_steps, vmax, center[lon,lat], bounds[[lon,lat],[lon,lat]], zoom}
  roads:  [ [[lon,lat,z],[lon,lat,z]], ... ]        # 3D segments
  buildings: GeoJSON FeatureCollection (properties.height)
  worlds: { name: {summary, trips_series, cars[], peds[]} }
    car:  {lng[], lat[], z[], hdg[], spd[], crash[]}   # hdg = rad CCW from east
    ped:  {lng[], lat[], z[]}
"""
from __future__ import annotations

import json
import os

import numpy as np

SCHEMA_VERSION = 1
_CAR_KEYS = {"lng", "lat", "z", "hdg", "spd", "crash", "arr"}
_META_KEYS = {"dt", "n_steps", "vmax", "center", "bounds"}
_WORLD_KEYS = {"summary", "trips_series", "cars", "peds"}
_SUMMARY_KEYS = {"cars", "crashed_end"}


def pack_world(*, car_lon, car_lat, car_z, heading, speed, crashed, goals,
               ped_lon, ped_lat, ped_z, stride: int, arrived=None) -> dict:
    """Reproject-agnostic packer: takes already-lon/lat arrays (T, N) -> world dict.

    `arrived` (T, N) bool marks cars that have reached their destination (finite
    cohort, remove-on-arrival). Both crash and arrival LATCH so the viewer can keep
    a car red (crashed) / green (arrived) / blue (en route) for the rest of the run.
    """
    T, N = car_lon.shape
    frames = range(0, T, stride)
    persist_crash = np.cumsum(crashed.astype(np.int32), axis=0) > 0
    if arrived is None:
        arrived = np.zeros((T, N), bool)
    persist_arr = np.cumsum(np.asarray(arrived).astype(np.int32), axis=0) > 0

    cars = []
    for i in range(N):
        cars.append({
            "lng": [round(float(car_lon[t, i]), 6) for t in frames],
            "lat": [round(float(car_lat[t, i]), 6) for t in frames],
            "z":   [round(float(car_z[t, i]), 2) for t in frames],
            "hdg": [round(float(heading[t, i]), 4) for t in frames],
            "spd": [round(float(speed[t, i]), 2) for t in frames],
            "crash": [int(persist_crash[t, i]) for t in frames],
            "arr": [int(persist_arr[t, i]) for t in frames],
        })

    peds = []
    for j in range(ped_lon.shape[1]):
        peds.append({
            "lng": [round(float(ped_lon[t, j]), 6) for t in frames],
            "lat": [round(float(ped_lat[t, j]), 6) for t in frames],
            "z":   [round(float(ped_z[t, j]), 2) for t in frames],
        })

    moving_end = int(((speed[-1] > 1.0) & ~persist_crash[-1]).sum())
    summary = {
        "cars": int(N), "peds": int(ped_lon.shape[1]),
        "trips_end": int(goals[-1].sum()),
        "crashed_end": int(persist_crash[-1].sum()),
        "arrived_end": int(persist_arr[-1].sum()),
        "moving_end": moving_end,
    }
    trips_series = [int(goals[t].sum()) for t in frames]
    return {"summary": summary, "trips_series": trips_series, "cars": cars, "peds": peds}


def build_scene(*, meta: dict, roads: list, buildings: dict, worlds: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "roads": roads,
        "buildings": buildings,
        "worlds": worlds,
    }


def validate_scene(scene: dict) -> None:
    """Raise ValueError if `scene` does not conform to schema v1.

    This is THE contract: any sim backend (kinematic now, Isaac/PhysX later) must
    emit a scene that passes here, and a scene that passes here must be renderable
    by the viewer. So we enforce every field the viewer actually reads — a loose
    validator that lets `meta: {}` through would fail silently in the browser.
    """
    if scene.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, "
                         f"got {scene.get('schema_version')!r}")
    for key in ("meta", "worlds"):
        if key not in scene:
            raise ValueError(f"scene missing required key: {key!r}")

    missing_meta = _META_KEYS - set(scene["meta"])
    if missing_meta:
        raise ValueError(f"meta missing keys: {sorted(missing_meta)}")

    if not scene["worlds"]:
        raise ValueError("scene has no worlds")

    for wname, world in scene["worlds"].items():
        missing_world = _WORLD_KEYS - set(world)
        if missing_world:
            raise ValueError(f"world {wname!r} missing keys: {sorted(missing_world)}")
        missing_summary = _SUMMARY_KEYS - set(world["summary"])
        if missing_summary:
            raise ValueError(f"world {wname!r} summary missing keys: "
                             f"{sorted(missing_summary)}")
        for car in world["cars"]:
            missing = _CAR_KEYS - set(car)
            if missing:
                raise ValueError(f"world {wname!r} car missing keys: {sorted(missing)}")
            n = len(car["lng"])
            uneven = {k: len(car[k]) for k in _CAR_KEYS if len(car[k]) != n}
            if uneven:
                raise ValueError(f"world {wname!r} car has uneven frame counts "
                                 f"(lng={n}): {uneven}")


def write_scene(path: str, scene: dict) -> int:
    """Validate then write compact JSON. Returns bytes written."""
    validate_scene(scene)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        # allow_nan=False: NaN/Infinity are invalid JSON (the browser's JSON.parse
        # rejects them). Fail loudly at export rather than write a scene the viewer
        # can't load.
        json.dump(scene, f, separators=(",", ":"), allow_nan=False)
    return os.path.getsize(path)
