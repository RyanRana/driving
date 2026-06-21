"""Regression: the graph cache key must depend on the bbox.

A fixed cache filename silently returned one region's graph for every bbox,
making `--region` a no-op (every region resolved to whatever was cached first).
"""
from smoothride.data.map_loader import SF_REGIONS, _bbox_cache_name


def test_distinct_regions_get_distinct_cache_names():
    names = {region: _bbox_cache_name(bbox) for region, bbox in SF_REGIONS.items()}
    # no two regions may collide on the same cache file
    assert len(set(names.values())) == len(names), names


def test_cache_name_is_deterministic_and_graphml():
    bbox = SF_REGIONS["mission"]
    n1 = _bbox_cache_name(bbox)
    n2 = _bbox_cache_name(bbox)
    assert n1 == n2
    assert n1.endswith(".graphml")


def test_downtown_and_mission_differ():
    assert _bbox_cache_name(SF_REGIONS["downtown"]) != _bbox_cache_name(
        SF_REGIONS["mission"]
    )
