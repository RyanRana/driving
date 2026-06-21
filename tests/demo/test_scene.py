import numpy as np
from smoothride.demo import scene as S


def test_pack_world_shapes_and_keys():
    T, N, M = 5, 2, 1
    world = S.pack_world(
        car_lon=np.zeros((T, N)), car_lat=np.zeros((T, N)),
        car_z=np.zeros((T, N)), heading=np.zeros((T, N)),
        speed=np.ones((T, N)), crashed=np.zeros((T, N), bool),
        goals=np.zeros((T, N), int),
        ped_lon=np.zeros((T, M)), ped_lat=np.zeros((T, M)), ped_z=np.zeros((T, M)),
        stride=1,
    )
    assert set(world) == {"summary", "trips_series", "cars", "peds"}
    assert len(world["cars"]) == N
    assert set(world["cars"][0]) == {"lng", "lat", "z", "hdg", "spd", "crash", "arr"}
    assert len(world["cars"][0]["lng"]) == T


def test_pack_world_arrived_latches_and_summarizes():
    # car arrives at frame 2 -> "arr" is 0 before, 1 from then on (latched);
    # summary reports the end-of-run arrived count.
    T, N = 4, 1
    arrived = np.zeros((T, N), bool)
    arrived[2:, 0] = True
    world = S.pack_world(
        car_lon=np.zeros((T, N)), car_lat=np.zeros((T, N)), car_z=np.zeros((T, N)),
        heading=np.zeros((T, N)), speed=np.zeros((T, N)),
        crashed=np.zeros((T, N), bool), goals=np.zeros((T, N), int), arrived=arrived,
        ped_lon=np.zeros((T, 0)), ped_lat=np.zeros((T, 0)), ped_z=np.zeros((T, 0)),
        stride=1)
    assert world["cars"][0]["arr"] == [0, 0, 1, 1]
    assert world["summary"]["arrived_end"] == 1


def test_validate_scene_accepts_minimal_valid_scene():
    scene = {
        "schema_version": S.SCHEMA_VERSION,
        "meta": {"dt": 0.2, "n_steps": 1, "vmax": 16.0,
                 "center": [-122.41, 37.79], "bounds": [[0, 0], [1, 1]], "zoom": 15},
        "roads": [[[-122.41, 37.79, 5.0], [-122.40, 37.79, 6.0]]],
        "buildings": {"type": "FeatureCollection", "features": []},
        "worlds": {"trained": S.pack_world(
            car_lon=np.zeros((1, 1)), car_lat=np.zeros((1, 1)), car_z=np.zeros((1, 1)),
            heading=np.zeros((1, 1)), speed=np.zeros((1, 1)),
            crashed=np.zeros((1, 1), bool), goals=np.zeros((1, 1), int),
            ped_lon=np.zeros((1, 0)), ped_lat=np.zeros((1, 0)), ped_z=np.zeros((1, 0)),
            stride=1)},
    }
    S.validate_scene(scene)      # must not raise


def test_validate_scene_rejects_wrong_version():
    import pytest
    with pytest.raises(ValueError):
        S.validate_scene({"schema_version": 999, "meta": {}, "worlds": {}})


def _minimal_world():
    return S.pack_world(
        car_lon=np.zeros((1, 1)), car_lat=np.zeros((1, 1)), car_z=np.zeros((1, 1)),
        heading=np.zeros((1, 1)), speed=np.zeros((1, 1)),
        crashed=np.zeros((1, 1), bool), goals=np.zeros((1, 1), int),
        ped_lon=np.zeros((1, 0)), ped_lat=np.zeros((1, 0)), ped_z=np.zeros((1, 0)),
        stride=1)


def _minimal_scene():
    return {
        "schema_version": S.SCHEMA_VERSION,
        "meta": {"dt": 0.2, "n_steps": 1, "vmax": 16.0,
                 "center": [-122.41, 37.79], "bounds": [[0, 0], [1, 1]], "zoom": 15},
        "roads": [],
        "buildings": {"type": "FeatureCollection", "features": []},
        "worlds": {"trained": _minimal_world()},
    }


def test_validate_scene_rejects_empty_meta():
    import pytest
    scene = _minimal_scene()
    scene["meta"] = {}                              # a future backend forgetting meta
    with pytest.raises(ValueError):
        S.validate_scene(scene)


def test_validate_scene_rejects_no_worlds():
    import pytest
    scene = _minimal_scene()
    scene["worlds"] = {}
    with pytest.raises(ValueError):
        S.validate_scene(scene)


def test_validate_scene_rejects_uneven_car_frames():
    import pytest
    scene = _minimal_scene()
    scene["worlds"]["trained"]["cars"][0]["z"] = []   # z shorter than lng
    with pytest.raises(ValueError):
        S.validate_scene(scene)


def test_pack_world_stride_decimates_frames():
    T, N = 6, 1
    world = S.pack_world(
        car_lon=np.zeros((T, N)), car_lat=np.zeros((T, N)), car_z=np.zeros((T, N)),
        heading=np.zeros((T, N)), speed=np.zeros((T, N)),
        crashed=np.zeros((T, N), bool), goals=np.zeros((T, N), int),
        ped_lon=np.zeros((T, 0)), ped_lat=np.zeros((T, 0)), ped_z=np.zeros((T, 0)),
        stride=2)
    assert len(world["cars"][0]["lng"]) == 3        # frames 0,2,4
    assert len(world["trips_series"]) == 3


def test_pack_world_crash_persists_after_event():
    T, N = 4, 1
    crashed = np.zeros((T, N), bool)
    crashed[1, 0] = True                            # crashes at step 1 only
    world = S.pack_world(
        car_lon=np.zeros((T, N)), car_lat=np.zeros((T, N)), car_z=np.zeros((T, N)),
        heading=np.zeros((T, N)), speed=np.zeros((T, N)),
        crashed=crashed, goals=np.zeros((T, N), int),
        ped_lon=np.zeros((T, 0)), ped_lat=np.zeros((T, 0)), ped_z=np.zeros((T, 0)),
        stride=1)
    assert world["cars"][0]["crash"] == [0, 1, 1, 1]   # latches on, stays on
    assert world["summary"]["crashed_end"] == 1
