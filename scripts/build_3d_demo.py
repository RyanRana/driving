"""One command, end to end: build the 3D ground truth, roll out a policy, write a
reproducible trace, run the deterministic verifier, and export the CesiumJS demo.

    python scripts/build_3d_demo.py                 # scripted driver, 40 cars
    python scripts/build_3d_demo.py --agents 80 --steps 400
    python scripts/build_3d_demo.py --ckpt runs/trained.msgpack

Then serve and open the viewer (printed at the end).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find preview_terrain

from smoothride.demo import export_cesium  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-preview", action="store_true",
                    help="skip the hillshade/footprints PNG")
    args = ap.parse_args()

    if not args.no_preview:
        print("== [1/2] terrain + buildings preview ==")
        try:
            import preview_terrain
            preview_terrain.main()
        except Exception as e:  # noqa: BLE001
            print(f"   preview skipped ({e!r})")

    print("== [2/2] 3D ground truth -> rollout -> trace -> verify -> export ==")
    argv = ["export_cesium", "--agents", str(args.agents), "--steps", str(args.steps)]
    if args.ckpt:
        argv += ["--ckpt", args.ckpt]
    sys.argv = argv
    export_cesium.main()

    print("\n== done. serve the demo: ==")
    print("   python -m http.server 8000")
    print("   open http://localhost:8000/smoothride/demo/cesium/")
    print("   (add a free Cesium ion token in smoothride/demo/cesium/config.js"
          " for terrain + 3D buildings)")


if __name__ == "__main__":
    main()
