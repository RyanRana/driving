"""Phase 1 — 3D ground truth: terrain z + edge grade + building footprints.

These touch the network on a cold cache; once data_cache/ is warm they're fast
and offline. Skipped cleanly if terrain couldn't be fetched at all.
"""
import numpy as np
import pytest

from smoothride.data.map_loader import load_road_network
from smoothride.data.terrain import add_terrain
from smoothride.data.buildings import load_building_set


@pytest.fixture(scope="module")
def net():
    n = load_road_network()
    add_terrain(n)
    return n


def test_node_z_plausible_for_sf(net):
    if not np.any(net.node_z):
        pytest.skip("terrain unavailable (offline)")
    z = net.node_z
    assert z.shape == (net.n_nodes,)
    # downtown SF sits ~0-280 m; allow a little slack for DEM noise.
    assert z.min() >= -5.0 and z.max() <= 300.0


def test_some_edges_are_steep(net):
    if not np.any(net.node_z):
        pytest.skip("terrain unavailable (offline)")
    # SF hills are real: at least one edge should exceed a 15% grade.
    assert np.abs(net.edge_grade).max() > 0.15


def test_grades_are_clipped(net):
    assert np.abs(net.edge_grade).max() <= 0.35 + 1e-6


def test_junction_nodes_single_valued(net):
    # node_z is indexed by node, so each shared junction node has exactly one z.
    assert net.node_z.shape == (net.n_nodes,)
    assert not np.isnan(net.node_z).any()


def test_buildings_load_and_lie_in_frame(net):
    bset = load_building_set(net)
    if bset.n_buildings == 0:
        pytest.skip("buildings unavailable (offline)")
    x0, y0, x1, y1 = net.bounds()
    # footprints reproject into the same metric frame, with generous margin.
    for ring in bset.polygons[:50]:
        assert ring.shape[1] == 2
        assert (ring[:, 0] > x0 - 200).all() and (ring[:, 0] < x1 + 200).all()
        assert (ring[:, 1] > y0 - 200).all() and (ring[:, 1] < y1 + 200).all()
    assert bset.segments.shape[1:] == (2, 2)
    assert (bset.height > 0).all()
