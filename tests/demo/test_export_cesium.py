import numpy as np
from smoothride.demo.export_cesium import sample_path_z


def test_sample_path_z_picks_nearest_node_elevation():
    # two nodes: (0,0,z=0) and (100,0,z=10). A point at (90,0) is nearest node 1.
    node_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    node_z = np.array([0.0, 10.0])
    pos = np.array([[[90.0, 0.0]]])          # (T=1, N=1, 2)
    z = sample_path_z(pos, node_xy, node_z)
    assert z.shape == (1, 1)
    np.testing.assert_allclose(z, [[10.0]])
