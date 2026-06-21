# Deploy (Vercel)

This is a static site (the Nomos 3D viewer + landing page). No build step.

- `vercel.json` (repo root) serves `smoothride/demo/` as the site root.
- `/`     → landing page (`landing.html`)
- `/sim`  → 3D simulation (`cesium/index.html`)

## Deploy
```
vercel --prod          # from the repo root; uses vercel.json
```
Or connect the GitHub repo in the Vercel dashboard — no settings needed
(framework: Other, no build command, output dir `smoothride/demo`).

The Cesium ion token is embedded in `cesium/app.js`, so 3D terrain + buildings
render with zero configuration. Rotate it at ion.cesium.com if needed.
