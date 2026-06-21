"""Shared fixtures for the RL-side (trace + verifier) tests.

Pure/offline — no JAX, no env, no network. Fabricate a small `Trace` by hand and
override only the fields a test cares about. Defaults describe a clean run: each car
sits on the lane-0 centerline of a straight +x segment, facing forward, stationary.
"""
from __future__ import annotations

import numpy as np
import pytest

from smoothride.rl.trace import Trace, TraceManifest


@pytest.fixture
def make_trace():
    def _make(n_steps: int = 3, n_agents: int = 2, n_peds: int = 2,
              dt: float = 0.2, collision_radius: float = 2.2,
              lane_width: float = 3.5,
              ped_pos: list | np.ndarray | None = None,
              ped_crossing: list | np.ndarray | None = None,
              **overrides):
        T, N, M = n_steps, n_agents, n_peds
        # Straight unit segment along +x; lane-0 centerline is offset right by
        # lane_width*0.5. right-normal of +x is (0,-1), so the centerline sits at
        # y = -lane_width*0.5. Place each car there → lateral offset 0 (on-lane).
        pos = np.zeros((T, N, 2), np.float32)
        pos[..., 1] = -lane_width * 0.5
        seg_start = np.zeros((T, N, 2), np.float32)
        seg_end = np.zeros((T, N, 2), np.float32)
        seg_end[..., 0] = 1.0
        # Default peds: parked far away and not crossing → ped-yield term contributes 0
        ped_pos_arr = (np.asarray(ped_pos, np.float32)
                       if ped_pos is not None
                       else np.full((T, M, 2), 1e6, np.float32))
        ped_crossing_arr = (np.asarray(ped_crossing, bool)
                            if ped_crossing is not None
                            else np.zeros((T, M), bool))
        fields = dict(
            pos=pos,
            z=np.zeros((T, N), np.float32),
            heading=np.zeros((T, N), np.float32),     # facing +x = route direction
            speed=np.zeros((T, N), np.float32),       # stationary → no wrong-way
            lane=np.zeros((T, N), np.int32),
            action=np.zeros((T, N, 3), np.float32),
            wp_ptr=np.zeros((T, N), np.int32),
            dist_remaining=np.zeros((T, N), np.float32),
            seg_start=seg_start,
            seg_end=seg_end,
            lane_count=np.ones((T, N), np.int32),
            spawn_grace=np.zeros((T, N), np.int32),
            crashed=np.zeros((T, N), bool),
            arrived=np.zeros((T, N), bool),
            speed_limit=np.full((T, N), 1e9, np.float32),
        )
        # Convert any list overrides to numpy arrays matching original field dtype
        for k, v in overrides.items():
            if isinstance(v, list):
                orig = fields.get(k)
                dtype = orig.dtype if orig is not None else np.float32
                overrides[k] = np.asarray(v, dtype)
        fields.update(overrides)
        manifest = TraceManifest(
            run_id="test-run", seed=0, scenario_id="test", policy_checkpoint_id="ckpt",
            config_hash="hash", dt=dt, n_steps=T, n_agents=N, n_peds=M,
        )
        return Trace(manifest=manifest, collision_radius=collision_radius,
                     lane_width=lane_width, ped_pos=ped_pos_arr,
                     ped_crossing=ped_crossing_arr, **fields)

    return _make
