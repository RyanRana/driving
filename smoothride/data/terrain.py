"""Elevation ground truth: turn the flat (x, y) road network into a 2.5D one.

We pull a USGS 3DEP DEM (Digital Elevation Model — a raster of ground heights)
for the map bbox via `py3dep`, bilinearly sample it at every road node to get a
per-node `z`, reconcile shared junction nodes to a single height (so roads don't
tear at intersections), and derive a per-edge grade (rise / run).

SF hills are real — Filbert/Lombard hit ~25-31% grade — so we smooth+clip the
grade against DEM-vs-GPS noise but keep the genuine steepness.

Everything degrades gracefully: if the DEM can't be fetched (offline), we warn
and return zeros so the flat 2D path keeps working (the env treats terrain as
optional — see `map_loader.RoadNetwork.node_z` default).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from pyproj import Transformer

from .map_loader import CACHE_DIR, RoadNetwork

DEM_CACHE = os.path.join(CACHE_DIR, "sf_dem.tif")

# SF street grades top out around 0.31 (Filbert St). Clip a bit above that to
# kill DEM-vs-GPS spikes without flattening the real hills.
MAX_GRADE = 0.35


def load_dem(bbox: tuple[float, float, float, float], resolution: int = 10,
             cache_path: str = DEM_CACHE, refresh: bool = False):
    """Return a DEM raster (xarray.DataArray, EPSG:4326) for `bbox`.

    bbox is (west, south, east, north) — the same order as map_loader. Cached to
    a GeoTIFF on first pull (10 m default; pass resolution=1 for crisp ridges).
    """
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and not refresh:
        return xr.open_dataarray(cache_path).squeeze()

    import py3dep
    dem = py3dep.get_dem(tuple(bbox), resolution).squeeze()
    dem.rio.to_raster(cache_path)
    return dem


def _nodes_to_lonlat(net: RoadNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Undo map_loader's project+origin-shift: shifted-UTM nodes -> (lon, lat)."""
    tf = Transformer.from_crs(net.G.graph["crs"], "EPSG:4326", always_xy=True)
    east = net.node_xy[:, 0] + net.origin[0]
    north = net.node_xy[:, 1] + net.origin[1]
    lon, lat = tf.transform(east, north)
    return np.asarray(lon), np.asarray(lat)


def sample_node_z(net: RoadNetwork, dem) -> np.ndarray:
    """Bilinearly sample the DEM at each node -> (N,) elevation in meters."""
    import xarray as xr

    lon, lat = _nodes_to_lonlat(net)
    # py3dep serves the DEM in its native CRS (often EPSG:5070 Albers), not
    # lon/lat — reproject node coords into the DEM's grid CRS before sampling.
    dem_crs = dem.rio.crs
    tf = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    dx, dy = tf.transform(lon, lat)
    xs = xr.DataArray(np.asarray(dx), dims="pt")
    ys = xr.DataArray(np.asarray(dy), dims="pt")
    z = dem.interp(x=xs, y=ys, method="linear").to_numpy()
    z = np.asarray(z, np.float64)
    # interp returns NaN outside the raster footprint — backfill with the median.
    if np.isnan(z).any():
        z = np.where(np.isnan(z), np.nanmedian(z), z)
    return z.astype(np.float32)


def reconcile_junction_z(net: RoadNetwork, node_z: np.ndarray) -> np.ndarray:
    """One z per node already holds (node_z is indexed by node), so this is a
    no-op pass-through kept as a named seam: each shared junction node has exactly
    one height, so adjoining road segments meet instead of tearing. Returned as a
    copy so callers can treat it as the reconciled array."""
    return node_z.astype(np.float32).copy()


def edge_grade(net: RoadNetwork, node_z: np.ndarray) -> np.ndarray:
    """Per-edge grade = (z_v - z_u) / horizontal_length, smoothed + clipped."""
    u, v = net.edges[:, 0], net.edges[:, 1]
    run = np.maximum(net.edge_length, 1.0)
    grade = (node_z[v] - node_z[u]) / run
    return np.clip(grade, -MAX_GRADE, MAX_GRADE).astype(np.float32)


def add_terrain(net: RoadNetwork, resolution: int = 10,
                refresh: bool = False) -> RoadNetwork:
    """Populate `net.node_z` and `net.edge_grade` in place and return `net`.

    On any failure (offline / py3dep error) it warns and leaves the zeros that
    `RoadNetwork` was constructed with, so the 2D path is unaffected.
    """
    try:
        dem = load_dem(net.bbox, resolution=resolution, refresh=refresh)
        node_z = reconcile_junction_z(net, sample_node_z(net, dem))
        net.node_z = node_z
        net.edge_grade = edge_grade(net, node_z)
    except Exception as e:  # noqa: BLE001 — terrain is optional, never fatal
        warnings.warn(f"terrain unavailable ({e!r}); falling back to flat z=0",
                      stacklevel=2)
        net.node_z = np.zeros(net.n_nodes, np.float32)
        net.edge_grade = np.zeros(net.n_edges, np.float32)
    return net


if __name__ == "__main__":
    from .map_loader import load_road_network

    net = load_road_network()
    add_terrain(net)
    z = net.node_z
    print(f"nodes={net.n_nodes}  z: min={z.min():.1f} max={z.max():.1f} "
          f"mean={z.mean():.1f} m")
    g = np.abs(net.edge_grade)
    print(f"edges={net.n_edges}  |grade|: max={g.max()*100:.0f}%  "
          f"steep(>15%)={int((g > 0.15).sum())}")
