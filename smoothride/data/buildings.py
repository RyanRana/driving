"""OSM building footprints -> extruded 3D GeoJSON for the Cesium demo.

Buildings are occluders + colliders + visual meshes. We pull footprint polygons
from OSM (same source as the roads), impute a height (explicit -> levels -> default),
and emit a lon/lat FeatureCollection with a `height` property the viewer extrudes.
"""
from __future__ import annotations

DEFAULT_HEIGHT = 8.0     # m, ~2-3 storeys when nothing is tagged
METERS_PER_LEVEL = 3.0


def impute_height(tags: dict) -> float:
    """explicit `height` -> `building:levels` * 3 m -> DEFAULT_HEIGHT."""
    h = tags.get("height")
    if h is not None:
        try:
            return float(str(h).split()[0])     # strip a trailing unit if present
        except (ValueError, IndexError):
            pass
    lvl = tags.get("building:levels")
    if lvl is not None:
        try:
            return float(lvl) * METERS_PER_LEVEL
        except (ValueError, TypeError):
            pass
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
