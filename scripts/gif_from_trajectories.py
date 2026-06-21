"""Render the demo trajectories.json straight to a GIF + start/mid/end PNGs.

The normal artifact path (smoothride.demo.render) replays a trained .msgpack
checkpoint. When you only have an exported trajectories.json (e.g. the lane-
geometric fleet from smoothride.demo.export_lanes) and no checkpoint, this reuses
the SAME renderer: it reprojects the lon/lat tracks back into the road network's
metric frame and calls demo.render.render(), so the output matches the project's
usual GIF/stills style.

  python scripts/gif_from_trajectories.py \
      --traj smoothride/demo/web/public/trajectories.json --name lane_demo
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pyproj import Transformer

from smoothride.data.map_loader import load_road_network
from smoothride.demo.render import render

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "artifacts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="smoothride/demo/web/public/trajectories.json")
    ap.add_argument("--name", default="lane_demo")
    ap.add_argument("--title", default="SmoothRide — lane-geometric fleet (SF)")
    ap.add_argument("--world", default="trained")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--zoom", type=float, default=1.0,
                    help=">1 crops to a centered pocket (e.g. 2.5) to show lanes/turns")
    args = ap.parse_args()

    data = json.load(open(args.traj))
    cars = data["worlds"][args.world]["cars"]
    N, T = len(cars), len(cars[0]["lng"])

    net = load_road_network()
    # lon/lat -> origin-shifted UTM meters (inverse of the export reprojection)
    to_m = Transformer.from_crs("EPSG:4326", net.G.graph["crs"], always_xy=True)
    lon = np.array([c["lng"] for c in cars]).T          # (T, N)
    lat = np.array([c["lat"] for c in cars]).T
    ex, ny = to_m.transform(lon.ravel(), lat.ravel())
    pos = np.stack([np.asarray(ex) - net.origin[0],
                    np.asarray(ny) - net.origin[1]], axis=-1).reshape(T, N, 2)

    crashed = np.array([c.get("crash", [0] * T) for c in cars]).T            # (T, N)
    # "trips done" counter: per-car arrival flag if present, else zeros
    goals = np.array([c.get("arr", [0] * T) for c in cars]).T
    ped = np.zeros((T, 0, 2), np.float32)               # this export carries no peds

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, args.name)
    gif = render(net, pos, crashed, goals, ped, out, args.title,
                 stride=args.stride, fps=args.fps)

    # optional zoom: crop the saved figure axes to a centered pocket so the
    # in-lane driving and clean turns are clearly visible.
    if args.zoom > 1.0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.collections import LineCollection
        cx = float(pos[..., 0].mean()); cy = float(pos[..., 1].mean())
        x0, y0, x1, y1 = net.bounds()
        hw = (x1 - x0) / (2 * args.zoom); hh = (y1 - y0) / (2 * args.zoom)
        fig, ax = plt.subplots(figsize=(8, 7), dpi=110)
        ax.set_facecolor("#0e1116"); fig.patch.set_facecolor("#0e1116")
        ax.add_collection(LineCollection(net.node_xy[net.edges], colors="#39404d", linewidths=1.4))
        ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hh, cy + hh)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(args.title + "  (zoom)", color="white", fontsize=13, pad=10)
        scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=80, c="#3b82f6",
                          edgecolors="white", linewidths=0.6, zorder=3)
        T = pos.shape[0]

        def upd(t):
            scat.set_offsets(pos[t])
            scat.set_color(np.where(crashed[t], "#ef4444", "#3b82f6"))
            return scat,
        anim = animation.FuncAnimation(fig, upd, frames=range(0, T, args.stride), blit=False)
        zgif = out + "_zoom.gif"
        anim.save(zgif, writer=animation.PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"saved zoom: {zgif}")
    print(f"cars={N} frames={T} trips_end={int(goals[-1].sum())} "
          f"crashed_end={int(crashed[-1].sum())}")
    print(f"saved: {gif} (+ _start/_mid/_end.png)")


if __name__ == "__main__":
    main()
