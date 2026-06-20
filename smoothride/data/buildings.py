"""OSM building footprints: occluders for 3D line-of-sight perception + render.

Buildings are community-mapped polygons in OpenStreetMap (`building=*`), same
ODbL source as the roads. We pull them with OSMnx, estimate a height (explicit
`height`, else `building:levels x 3 m`, else a default), reproject the footprints
into the road net's shifted-UTM metric frame, and produce two products:

  * `polygons` + `base_z` + `height` — for any host-side render / preview.
  * `segments` (S, 2, 2) — every footprint edge flattened into a wall segment,
    the occluder set the JAX perception layer ray-casts against (a building hides
    the car behind it).

The Cesium demo draws real 3D buildings from Cesium OSM Buildings (browser side),
so these are primarily the *simulation* occluders, not the render geometry.

Degrades gracefully: if the OSM pull fails (offline), returns an empty BuildingSet
so perception simply has no occluders (every neighbor visible).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

from .map_loader import CACHE_DIR, RoadNetwork

BUILDINGS_CACHE = os.path.join(CACHE_DIR, "sf_buildings.parquet")
DEFAULT_HEIGHT = 8.0     # m, when neither height nor levels is tagged
LEVEL_HEIGHT = 3.0       # m per storey


@dataclass
class BuildingSet:
    polygons: list[np.ndarray]   # B footprints, each (Vi, 2) metric exterior ring
    base_z: np.ndarray           # (B,) ground elevation at the footprint, meters
    height: np.ndarray           # (B,) building height, meters
    segments: np.ndarray         # (S, 2, 2) occluder wall segments, metric frame

    @property
    def n_buildings(self) -> int:
        return len(self.polygons)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @staticmethod
    def empty() -> "BuildingSet":
        return BuildingSet(polygons=[], base_z=np.zeros(0, np.float32),
                           height=np.zeros(0, np.float32),
                           segments=np.zeros((0, 2, 2), np.float32))


def load_buildings(bbox, cache_path: str = BUILDINGS_CACHE, refresh: bool = False):
    """Return a GeoDataFrame of building footprints for `bbox` (cached to parquet)."""
    import geopandas as gpd

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and not refresh:
        return gpd.read_parquet(cache_path)

    import osmnx as ox
    gdf = ox.features_from_bbox(tuple(bbox), tags={"building": True})
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    # Keep only columns we can serialize cleanly to parquet.
    keep = [c for c in ("geometry", "height", "building:levels") if c in gdf.columns]
    gdf = gdf[keep]
    gdf.to_parquet(cache_path)
    return gdf


def _num(val) -> float | None:
    """Parse OSM numeric-ish tags ('12', '12 m', '12.5') -> float, else None.
    Treats pandas NaN / missing as None."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(str(val).split()[0])
    except (ValueError, IndexError):
        return None


def height_of(row) -> float:
    """Building height: explicit `height`, else `levels x 3 m`, else default."""
    h = _num(row.get("height"))
    if h is not None and h > 0:
        return h
    lv = _num(row.get("building:levels"))
    if lv is not None and lv > 0:
        return lv * LEVEL_HEIGHT
    return DEFAULT_HEIGHT


def _base_z(net: RoadNetwork, centroids: np.ndarray) -> np.ndarray:
    """Ground elevation under each footprint = z of the nearest road node."""
    if net.node_z.size == 0 or not np.any(net.node_z):
        return np.zeros(len(centroids), np.float32)
    d2 = ((centroids[:, None, :] - net.node_xy[None, :, :]) ** 2).sum(-1)
    return net.node_z[d2.argmin(1)].astype(np.float32)


def to_metric_polygons(net: RoadNetwork, gdf) -> BuildingSet:
    """Reproject footprints into the road net's metric frame -> BuildingSet."""
    from shapely.geometry import Polygon

    gdf = gdf.to_crs(net.G.graph["crs"])
    ox0, oy0 = net.origin

    polygons: list[np.ndarray] = []
    heights: list[float] = []
    centroids: list[np.ndarray] = []
    seg_list: list[np.ndarray] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        h = height_of(row)
        for poly in parts:
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            ring = np.asarray(poly.exterior.coords, np.float32)[:, :2]
            ring = ring - np.array([ox0, oy0], np.float32)
            if len(ring) < 3:
                continue
            polygons.append(ring)
            heights.append(h)
            centroids.append(ring.mean(0))
            # each consecutive pair (closed ring) is one occluder wall segment
            seg = np.stack([ring[:-1], ring[1:]], axis=1)   # (Vi-1, 2, 2)
            seg_list.append(seg)

    if not polygons:
        return BuildingSet.empty()

    centroids = np.asarray(centroids, np.float32)
    segments = np.concatenate(seg_list, axis=0).astype(np.float32)
    return BuildingSet(
        polygons=polygons,
        base_z=_base_z(net, centroids),
        height=np.asarray(heights, np.float32),
        segments=segments,
    )


def load_building_set(net: RoadNetwork, refresh: bool = False) -> BuildingSet:
    """Pull + reproject buildings for `net`'s bbox. Empty set on failure."""
    try:
        gdf = load_buildings(net.bbox, refresh=refresh)
        return to_metric_polygons(net, gdf)
    except Exception as e:  # noqa: BLE001 — buildings are optional
        warnings.warn(f"buildings unavailable ({e!r}); no occluders",
                      stacklevel=2)
        return BuildingSet.empty()


if __name__ == "__main__":
    from .map_loader import load_road_network
    from .terrain import add_terrain

    net = load_road_network()
    add_terrain(net)
    bset = load_building_set(net)
    print(f"buildings={bset.n_buildings}  occluder segments={bset.n_segments}")
    if bset.n_buildings:
        print(f"height: min={bset.height.min():.1f} max={bset.height.max():.1f} "
              f"mean={bset.height.mean():.1f} m")
        print(f"base z: min={bset.base_z.min():.1f} max={bset.base_z.max():.1f} m")
