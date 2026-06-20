"""Phase-1 gate artifact: a hillshade of the SF terrain with road network +
building footprints overlaid, saved as a PNG. Proves the 3D ground truth loaded.

    python scripts/preview_terrain.py            # -> runs/artifacts/terrain_preview.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from smoothride.data.map_loader import load_road_network  # noqa: E402
from smoothride.data.terrain import add_terrain  # noqa: E402
from smoothride.data.buildings import load_building_set  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs",
                                   "artifacts", "terrain_preview.png"))


def main():
    net = load_road_network()
    add_terrain(net)
    bset = load_building_set(net)
    x0, y0, x1, y1 = net.bounds()

    fig, ax = plt.subplots(figsize=(9, 8), dpi=110)
    fig.patch.set_facecolor("#0e1116")
    ax.set_facecolor("#0e1116")

    # node elevation as a colored scatter (cheap "hillshade")
    sc = ax.scatter(net.node_xy[:, 0], net.node_xy[:, 1], c=net.node_z,
                    cmap="terrain", s=14, zorder=3)
    fig.colorbar(sc, ax=ax, label="elevation (m)", shrink=0.7)

    # roads, colored by grade magnitude
    segs = net.node_xy[net.edges]
    g = np.abs(net.edge_grade)
    lc = LineCollection(segs, array=g, cmap="inferno", linewidths=1.4, zorder=2)
    ax.add_collection(lc)

    # building footprints
    if bset.n_buildings:
        ax.add_collection(PolyCollection(bset.polygons, facecolors="#5b6577",
                                         edgecolors="none", alpha=0.45, zorder=1))

    ax.set_xlim(x0 - 20, x1 + 20)
    ax.set_ylim(y0 - 20, y1 + 20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"SF 3D ground truth — z {net.node_z.min():.0f}-{net.node_z.max():.0f} m, "
        f"max grade {g.max()*100:.0f}%, {bset.n_buildings} buildings",
        color="white", fontsize=12, pad=10)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
