"""Self-contained demo GIF maker — turns a lane trajectories.json into animated
GIFs (full city + zoomed pocket) with a live trips/crashes/speed HUD.

No dependency on the (merge-churned) viewer or demo.render: it only needs the
road network for the street background and matplotlib's Pillow writer. The GIFs
loop and animate in VS Code's built-in image preview.

  python scripts/make_demo_gif.py --traj runs/lane_traj.json --name lane_demo
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "artifacts"))


def load(traj, world):
    d = json.load(open(traj))
    cars = d["worlds"][world]["cars"]
    meta = d["meta"]
    lon = np.array([c["lng"] for c in cars]).T            # (T, N)
    lat = np.array([c["lat"] for c in cars]).T
    spd = np.array([c.get("spd", [[0]] * len(cars))[0] and c["spd"] for c in cars]).T \
        if "spd" in cars[0] else np.zeros_like(lon)
    crash = np.array([c.get("crash", [0] * lon.shape[0]) for c in cars]).T
    arr = np.array([c.get("arr", [0] * lon.shape[0]) for c in cars]).T
    trips = meta.get("trips_series") or list(arr.sum(axis=1).astype(int))
    return meta, lon, lat, spd, crash, arr, trips


def road_segments():
    """(E,2,2) metric road segments + bounds, or None if the graph won't load."""
    try:
        from smoothride.data.map_loader import load_road_network
        net = load_road_network()
        return net.node_xy[net.edges], net.bounds()
    except Exception as e:
        print(f"(road background unavailable: {e})")
        return None, None


def to_metric(lon, lat):
    """lon/lat -> a local meter frame for plotting (equirectangular about the mean)."""
    lat0 = float(np.mean(lat))
    mlon = 111320.0 * np.cos(np.radians(lat0))
    return lon * mlon, lat * 111320.0


def make_gif(out, title, X, Y, spd, crash, trips, dt, fps, stride,
             segs_xy=None, zoom=None):
    T, N = X.shape
    fig, ax = plt.subplots(figsize=(9, 8), dpi=110)
    ax.set_facecolor("#0e1116"); fig.patch.set_facecolor("#0e1116")
    if segs_xy is not None:
        ax.add_collection(LineCollection(segs_xy, colors="#39404d",
                                         linewidths=1.6 if zoom else 0.9))
    if zoom:
        cx, cy = float(X.mean()), float(Y.mean())
        ax.set_xlim(cx - zoom, cx + zoom); ax.set_ylim(cy - zoom, cy + zoom)
    else:
        ax.set_xlim(X.min() - 40, X.max() + 40); ax.set_ylim(Y.min() - 40, Y.max() + 40)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color="white", fontsize=14, pad=12)
    hud = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", color="white",
                  fontsize=11, family="monospace",
                  bbox=dict(boxstyle="round", fc="#0c0f14cc", ec="#2a313d"))
    size = 130 if zoom else 34
    scat = ax.scatter(X[0], Y[0], s=size, c="#3b82f6",
                      edgecolors="white", linewidths=0.5, zorder=3)

    def upd(t):
        scat.set_offsets(np.c_[X[t], Y[t]])
        scat.set_color(np.where(crash[t] > 0, "#ef4444", "#3b82f6"))
        moving = int((spd[t] > 0.5).sum())
        mph = float(spd[t][spd[t] > 0.5].mean()) * 2.23694 if moving else 0.0
        hud.set_text(f" cars {N}   trips {trips[t]:>3}   crashes {int((crash[t]>0).sum())}\n"
                     f" moving {moving:>3}   avg {mph:4.1f} mph   t {t*dt:4.1f}s")
        return scat, hud

    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim = animation.FuncAnimation(fig, upd, frames=range(0, T, stride), blit=False)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/lane_traj.json")
    ap.add_argument("--name", default="lane_demo")
    ap.add_argument("--world", default="trained")
    ap.add_argument("--title", default="SmoothRide — San Francisco fleet")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    meta, lon, lat, spd, crash, arr, trips = load(args.traj, args.world)
    segs, _ = road_segments()
    X, Y = to_metric(lon, lat)
    segs_xy = None
    if segs is not None:
        # reproject road nodes into the SAME local meter frame as the cars
        from smoothride.data.map_loader import load_road_network
        from pyproj import Transformer
        net = load_road_network()
        tf = Transformer.from_crs(net.G.graph["crs"], "EPSG:4326", always_xy=True)
        flat = segs.reshape(-1, 2)
        slon, slat = tf.transform(flat[:, 0] + net.origin[0], flat[:, 1] + net.origin[1])
        lat0 = float(np.mean(lat))
        sx = np.asarray(slon) * 111320.0 * np.cos(np.radians(lat0))
        sy = np.asarray(slat) * 111320.0
        segs_xy = np.stack([sx, sy], -1).reshape(-1, 2, 2)

    dt = meta["dt"]
    full = make_gif(os.path.join(OUT, args.name + ".gif"), args.title,
                    X, Y, spd, crash, trips, dt, args.fps, args.stride, segs_xy)
    print("saved", full)
    zoom = make_gif(os.path.join(OUT, args.name + "_zoom.gif"), args.title + "  (zoom)",
                    X, Y, spd, crash, trips, dt, args.fps, max(1, args.stride - 1),
                    segs_xy, zoom=260.0)
    print("saved", zoom)


if __name__ == "__main__":
    main()
