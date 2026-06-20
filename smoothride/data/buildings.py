"""OSM building footprints -> extruded 3D GeoJSON for the Cesium demo.

Buildings are occluders + colliders + visual meshes. We pull footprint polygons
from OSM (same source as the roads), impute a height (explicit -> levels -> default),
and emit a lon/lat FeatureCollection with a `height` property the viewer extrudes.
"""
from __future__ import annotations

import math

DEFAULT_HEIGHT = 8.0     # m, ~2-3 storeys when nothing is tagged
METERS_PER_LEVEL = 3.0


def _finite_positive(value) -> float | None:
    """Parse to a finite, positive float, else None.

    OSMnx returns missing tag columns as float NaN (not None), so a plain
    `is not None` check lets NaN through -> it serializes to invalid JSON and
    breaks the viewer. Anything non-finite or <= 0 is treated as "not tagged".
    """
    try:
        f = float(str(value).split()[0])        # strip a trailing unit if present
    except (ValueError, IndexError, TypeError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def impute_height(tags: dict) -> float:
    """explicit `height` -> `building:levels` * 3 m -> DEFAULT_HEIGHT.

    Always returns a finite, positive height (never NaN/0/negative).
    """
    h = _finite_positive(tags.get("height"))
    if h is not None:
        return h
    lvl = _finite_positive(tags.get("building:levels"))
    if lvl is not None:
        return lvl * METERS_PER_LEVEL
    return DEFAULT_HEIGHT


def extrude_ring(ring_lonlat: list[tuple[float, float]], height: float) -> list[list[float]]:
    """A lon/lat ring -> a closed list of [lon, lat, height] triples."""
    out = [[float(lon), float(lat), float(height)] for lon, lat in ring_lonlat]
    if out and out[0] != out[-1]:
        out.append(out[0])      # close the ring
    return out


def fetch_buildings_geojson(bbox, transformer=None) -> dict:
    """Pull OSM buildings for bbox and return an extruded lon/lat FeatureCollection.

    NETWORK CALL (OSMnx). bbox is OSMnx 2.x order (west, south, east, north).
    Returned features carry properties.height (m) for client-side extrusion.
    """
    import osmnx as ox
    gdf = ox.features_from_bbox(bbox, tags={"building": True})
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        height = impute_height(row.to_dict())
        for poly in polys:
            ring = list(poly.exterior.coords)        # already lon/lat from OSM
            features.append({
                "type": "Feature",
                "properties": {"height": height},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [extrude_ring(ring, height)],
                },
            })
    return {"type": "FeatureCollection", "features": features}
