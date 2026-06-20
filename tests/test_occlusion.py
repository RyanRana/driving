"""Phase 3 — building occlusion. A wall between two cars hides the far one; remove
it and the car reappears; the vectorized test matches a brute-force reference.
"""
import jax.numpy as jnp
import numpy as np

from smoothride.env import perception


def test_wall_between_two_cars_blocks_sightline():
    ego = jnp.array([[0.0, 0.0]])
    nbr = jnp.array([[[10.0, 0.0]]])           # neighbor 10 m straight ahead
    wall = jnp.array([[[5.0, -3.0], [5.0, 3.0]]])  # vertical wall crossing the path
    blocked = perception.occlusion_mask(ego, nbr, wall)
    assert bool(blocked[0, 0]) is True


def test_no_wall_means_visible():
    ego = jnp.array([[0.0, 0.0]])
    nbr = jnp.array([[[10.0, 0.0]]])
    blocked = perception.occlusion_mask(ego, nbr, jnp.zeros((0, 2, 2)))
    assert bool(blocked[0, 0]) is False


def test_wall_beside_path_does_not_block():
    ego = jnp.array([[0.0, 0.0]])
    nbr = jnp.array([[[10.0, 0.0]]])
    wall = jnp.array([[[5.0, 3.0], [5.0, 9.0]]])  # off to the side, no crossing
    blocked = perception.occlusion_mask(ego, nbr, wall)
    assert bool(blocked[0, 0]) is False


def _brute_blocked(ego, nbr, segs):
    """Reference: does segment ego->nbr intersect any wall? (python loop)."""
    def seg_int(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
        d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
        d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
        return (d1 * d2 < 0) and (d3 * d4 < 0)
    return any(seg_int(ego, nbr, s[0], s[1]) for s in segs)


def test_vectorized_matches_brute_force():
    rng = np.random.default_rng(0)
    N, K, S = 6, 3, 12
    ego = rng.uniform(-20, 20, (N, 2)).astype(np.float32)
    nbr = rng.uniform(-20, 20, (N, K, 2)).astype(np.float32)
    a = rng.uniform(-20, 20, (S, 2)).astype(np.float32)
    b = a + rng.uniform(-8, 8, (S, 2)).astype(np.float32)
    segs = np.stack([a, b], axis=1).astype(np.float32)

    vec = np.asarray(perception.occlusion_mask(
        jnp.asarray(ego), jnp.asarray(nbr), jnp.asarray(segs)))
    for i in range(N):
        for k in range(K):
            ref = _brute_blocked(ego[i], nbr[i, k], segs)
            assert bool(vec[i, k]) == ref, (i, k)
