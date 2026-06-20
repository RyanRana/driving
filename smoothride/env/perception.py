"""3D perception primitives: building occlusion + nearest-building distance.

The behavioral env is 2.5D (cars drive in the (x,y) plane draped onto terrain),
so line-of-sight occlusion is a 2D ray-vs-wall test: a neighbor is *occluded* if
the segment ego->neighbor crosses any building footprint edge. Buildings are tall
enough that a footprint crossing == a blocked sightline, which is what we want —
the policy genuinely can't see the car hidden behind a building.

All functions are pure JAX (vmap/jit friendly). Shapes:
  ego  : (N, 2)        car positions
  nbr  : (N, K, 2)     candidate neighbor positions (ego frame: world coords)
  segs : (S, 2, 2)     building wall segments [[x0,y0],[x1,y1]]
"""
from __future__ import annotations

import jax.numpy as jnp


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def occlusion_mask(ego: jnp.ndarray, nbr: jnp.ndarray,
                   segs: jnp.ndarray) -> jnp.ndarray:
    """(N, K) bool: True where the ego->neighbor sightline is blocked by a wall.

    Standard 2D segment-segment intersection (orientation test) between each
    sightline A->B and each building wall C->D, reduced over walls. Strict-sign
    comparison ignores exact endpoint/collinear touches (a non-issue here)."""
    if segs.shape[0] == 0:
        return jnp.zeros(nbr.shape[:2], bool)

    A = ego[:, None, None, :]          # (N,1,1,2)  sightline start
    B = nbr[:, :, None, :]             # (N,K,1,2)  sightline end
    C = segs[None, None, :, 0, :]      # (1,1,S,2)  wall start
    D = segs[None, None, :, 1, :]      # (1,1,S,2)  wall end

    dCD = D - C
    dBA = B - A
    # orientation of A,B relative to wall CD and of C,D relative to sightline AB
    d1 = _cross(dCD[..., 0], dCD[..., 1], (A - C)[..., 0], (A - C)[..., 1])
    d2 = _cross(dCD[..., 0], dCD[..., 1], (B - C)[..., 0], (B - C)[..., 1])
    d3 = _cross(dBA[..., 0], dBA[..., 1], (C - A)[..., 0], (C - A)[..., 1])
    d4 = _cross(dBA[..., 0], dBA[..., 1], (D - A)[..., 0], (D - A)[..., 1])
    crosses = (d1 * d2 < 0) & (d3 * d4 < 0)     # (N,K,S)
    return jnp.any(crosses, axis=-1)


def nearest_building_dist(ego: jnp.ndarray, segs: jnp.ndarray,
                          far: float = 50.0) -> jnp.ndarray:
    """(N,) distance from each car to the closest building wall (clipped to far)."""
    if segs.shape[0] == 0:
        return jnp.full((ego.shape[0],), far)

    p = ego[:, None, :]                # (N,1,2)
    a = segs[None, :, 0, :]            # (1,S,2)
    b = segs[None, :, 1, :]            # (1,S,2)
    ab = b - a
    t = jnp.clip(jnp.sum((p - a) * ab, -1) /
                 (jnp.sum(ab * ab, -1) + 1e-9), 0.0, 1.0)
    proj = a + t[..., None] * ab       # (N,S,2) closest point on each wall
    d = jnp.linalg.norm(p - proj, axis=-1)
    return jnp.minimum(d.min(axis=1), far)
