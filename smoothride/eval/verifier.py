"""Deterministic, trace-based run verifier — the source of truth for reward/validity.

PURE FUNCTION over a `Trace`: no re-simulation, no randomness, no wall-clock, no
network/LLM. Same trace -> same verdict, always. (GPU physics and float ordering
aren't reproducible across hardware, so a verifier that re-ran the sim could
disagree with itself; geometric predicates over logged numbers cannot.)

Predicates (each a deterministic boolean over numbers already in the trace):
  crash          min pairwise footprint distance < collision_radius at any step
  off_road       car center outside the drivable road polygon (if supplied),
                 else the trace's recorded off-road channel
  rule_violation speed exceeds the edge speed limit (beyond a small tolerance)
  arrived        reached its destination waypoint within the horizon
  valid_run      every constraint satisfied for every car at every step
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env.trace import Trace

SPEED_TOL = 0.5   # m/s slack on the speed-limit check (integration noise)


@dataclass
class CarVerdict:
    id: int
    arrived: bool
    travel_time: float            # seconds to first arrival (or horizon if none)
    valid: bool
    costs: dict = field(default_factory=dict)   # crash / off_road / rule counts


@dataclass
class RunVerdict:
    valid_run: bool
    n_cars: int
    trips: int                    # total arrivals across all cars
    throughput: float             # trips per (car * sim-minute)
    mean_travel_time: float
    median_travel_time: float
    crash_count: int
    offroad_count: int
    rule_count: int
    cars: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"kind": "summary", "valid_run": self.valid_run,
                "n_cars": self.n_cars, "trips": self.trips,
                "throughput": round(self.throughput, 4),
                "mean_travel_time": round(self.mean_travel_time, 2),
                "median_travel_time": round(self.median_travel_time, 2),
                "crash_count": self.crash_count,
                "offroad_count": self.offroad_count,
                "rule_count": self.rule_count}


def _pairwise_crash(x, y, radius) -> np.ndarray:
    """(T, N) bool: car involved in a footprint overlap at step t."""
    T, N = x.shape
    pos = np.stack([x, y], axis=-1)                 # (T, N, 2)
    diff = pos[:, :, None, :] - pos[:, None, :, :]  # (T, N, N, 2)
    d = np.linalg.norm(diff, axis=-1)               # (T, N, N)
    eye = np.eye(N, dtype=bool)[None]
    d = np.where(eye, np.inf, d)
    return d.min(axis=2) < radius


def _offroad_from_polygon(x, y, road_polygon) -> np.ndarray:
    """(T, N) bool via point-in-polygon against the drivable road polygon."""
    from shapely.geometry import Point
    from shapely.prepared import prep
    pp = prep(road_polygon)
    T, N = x.shape
    out = np.zeros((T, N), bool)
    for t in range(T):
        for i in range(N):
            out[t, i] = not pp.contains(Point(float(x[t, i]), float(y[t, i])))
    return out


def verify(trace: Trace, road_polygon=None) -> RunVerdict:
    tl = trace.timeline
    x, y = tl["x"], tl["y"]
    T, N = x.shape
    dt = trace.dt

    crash = _pairwise_crash(x, y, trace.collision_radius)        # (T,N)
    if road_polygon is not None:
        off_road = _offroad_from_polygon(x, y, road_polygon)
    else:
        off_road = tl["off_road"] > 0.5
    rule = tl["speed"] > (tl["speed_limit"] + SPEED_TOL)
    arrived_evt = tl["arrived"] > 0.5

    cars: list[CarVerdict] = []
    travel_times: list[float] = []
    for i in range(N):
        arr_steps = np.flatnonzero(arrived_evt[:, i])
        arrived = arr_steps.size > 0
        tt = float((arr_steps[0] + 1) * dt) if arrived else float(T * dt)
        if arrived:
            travel_times.append(tt)
        costs = {"crash": int(crash[:, i].sum()),
                 "off_road": int(off_road[:, i].sum()),
                 "rule": int(rule[:, i].sum())}
        valid = sum(costs.values()) == 0
        cars.append(CarVerdict(id=i, arrived=arrived, travel_time=round(tt, 2),
                               valid=valid, costs=costs))

    trips = int(arrived_evt.sum())
    sim_minutes = max(T * dt / 60.0, 1e-9)
    return RunVerdict(
        valid_run=all(c.valid for c in cars),
        n_cars=N, trips=trips,
        throughput=trips / (N * sim_minutes),
        mean_travel_time=float(np.mean(travel_times)) if travel_times else 0.0,
        median_travel_time=float(np.median(travel_times)) if travel_times else 0.0,
        crash_count=int(crash.sum()),
        offroad_count=int(off_road.sum()),
        rule_count=int(rule.sum()),
        cars=cars,
    )
