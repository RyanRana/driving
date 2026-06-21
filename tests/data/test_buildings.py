import numpy as np
from smoothride.data import buildings as B


def test_impute_height_from_explicit():
    assert B.impute_height({"height": "12.5"}) == 12.5


def test_impute_height_from_levels():
    assert B.impute_height({"building:levels": "3"}) == 9.0   # 3 * 3.0


def test_impute_height_default_when_missing():
    assert B.impute_height({}) == B.DEFAULT_HEIGHT


def test_impute_height_treats_nan_tags_as_missing():
    # OSMnx returns missing tag columns as float NaN, not None. NaN must NOT
    # leak through as the height (it serializes to invalid JSON and breaks the viewer).
    nan = float("nan")
    assert B.impute_height({"height": nan}) == B.DEFAULT_HEIGHT
    assert B.impute_height({"building:levels": nan}) == B.DEFAULT_HEIGHT
    assert B.impute_height({"height": nan, "building:levels": "4"}) == 12.0


def test_impute_height_is_always_finite_positive():
    import math
    for tags in ({"height": "0"}, {"height": "-5"}, {"height": "nan"}, {"building:levels": "0"}):
        h = B.impute_height(tags)
        assert math.isfinite(h) and h > 0


def test_extrude_ring_makes_closed_3d_loop():
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    out = B.extrude_ring(ring, height=9.0)
    assert out[0] == out[-1]                 # closed
    assert all(len(p) == 3 for p in out)     # (lon, lat, z) triples
    assert all(p[2] == 9.0 for p in out)     # roof height baked in
