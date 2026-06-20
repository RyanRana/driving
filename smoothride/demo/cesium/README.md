# SmoothRide — CesiumJS 3D viewer

Meshed cars driving the **real 3D San Francisco** — Cesium World Terrain (the
actual hills) + Cesium OSM Buildings — replaying a logged rollout with a
time-dynamic clock. The HUD is driven by the deterministic verifier.

This is the headline 3D deliverable. The 2D deck.gl viewer in `../web/` stays as
the always-works fallback.

## Run it

1. **Generate the trajectory data** (3D ground truth → rollout → trace → verify →
   export). From the repo root:
   ```bash
   pip install -e ".[data,rl]"
   python -m smoothride.demo.export_cesium
   ```
   This writes `public/trajectories.json` (and a reproducible trace under
   `runs/traces/`). By default it uses a scripted waypoint-follower so the cars
   drive the streets; pass `--ckpt runs/trained.msgpack` to replay a trained
   policy.

2. **Add a Cesium ion token** (free) so the terrain + 3D buildings load:
   - Get one at https://ion.cesium.com/tokens
   - `cp config.example.js config.js` and paste it in.
   - (Or pass it in the URL: `?token=...`.)

3. **Serve the folder** (a static server is required — `fetch` won't work over
   `file://`). From the repo root:
   ```bash
   python -m http.server 8000
   ```
   then open **http://localhost:8000/smoothride/demo/cesium/**

Without a token the viewer still runs, but on a flat globe (no terrain/buildings)
— it shows a one-line note and dismisses it.

## What you're seeing

- **Box meshes** = cars, colored by speed (red = stopped → green = fast),
  **red** = crashed. They climb the real SF grades because each pose is draped
  onto Cesium terrain.
- **HUD** = live moving/crashed counts + the run's verifier verdict
  (trips completed, `valid run` flag).

## Files

| file | role |
|---|---|
| `index.html` | page + HUD + CesiumJS CDN |
| `app.js` | viewer, terrain draping, time-dynamic cars, HUD |
| `config.example.js` | copy → `config.js`, paste your ion token |
| `public/trajectories.json` | exported per-car poses + verifier summary |
