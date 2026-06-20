// SmoothRide — CesiumJS 3D viewer.
// Replays exported trajectories (smoothride.demo.export_cesium) as meshed cars
// driving the real 3D San Francisco: Cesium World Terrain (hills) + Cesium OSM
// Buildings, with a time-dynamic clock. Heights are draped onto Cesium terrain so
// the cars climb the actual SF grades. HUD is driven by the deterministic verifier.

// --- tweakables ---------------------------------------------------------
const CAR = { length: 4.6, width: 2.0, height: 1.5 };  // meters
const CAR_Z_OFFSET = 0.9;     // lift the box so it sits ON the road, not in it
const GEOID = 32.5;           // ortho->ellipsoid fallback when terrain sampling is off
const PLAYBACK = 1.0;         // clock multiplier (1 = real time)
// -----------------------------------------------------------------------

const TOKEN =
  new URLSearchParams(location.search).get("token") ||
  (window.CESIUM_ION_TOKEN && window.CESIUM_ION_TOKEN !== "PASTE_YOUR_CESIUM_ION_TOKEN_HERE"
    ? window.CESIUM_ION_TOKEN : null);

function fail(html) {
  const el = document.getElementById("err");
  el.style.display = "flex";
  el.innerHTML = `<div class="box">${html}</div>`;
}

const C = window.Cesium;

async function boot() {
  if (typeof C === "undefined") {
    return fail(`<h2>CesiumJS failed to load</h2><p>The CDN script didn't load —
      check your network or the <code>&lt;script&gt;</code> tag in index.html.</p>`);
  }

  let data;
  try {
    data = await (await fetch("public/trajectories.json")).json();
  } catch (e) {
    return fail(`<h2>No trajectory data</h2>
      <p>Could not load <code>public/trajectories.json</code>. Generate it:</p>
      <p><code>python -m smoothride.demo.export_cesium</code></p>
      <p>then serve and open this folder (see the README).</p>`);
  }

  const hasToken = !!TOKEN;
  if (hasToken) C.Ion.defaultAccessToken = TOKEN;

  const viewer = new C.Viewer("cesium", {
    terrain: hasToken ? C.Terrain.fromWorldTerrain() : undefined,
    animation: true, timeline: true, baseLayerPicker: false, geocoder: false,
    homeButton: false, sceneModePicker: false, navigationHelpButton: false,
    fullscreenButton: false, infoBox: false, selectionIndicator: false,
  });
  viewer.scene.globe.depthTestAgainstTerrain = true;

  if (hasToken) {
    try {
      const osm = await C.createOsmBuildingsAsync();
      viewer.scene.primitives.add(osm);
    } catch (e) { console.warn("OSM buildings unavailable:", e); }
  } else {
    fail(`<h2>No Cesium ion token</h2>
      <p>Running without <b>terrain</b> or <b>3D buildings</b>. The cars still
      drive, but on a flat globe.</p>
      <p>Add a free token in <code>config.js</code> (copy
      <code>config.example.js</code>) to see the real 3D San Francisco.</p>
      <p style="opacity:.6">This message auto-dismisses in 6s.</p>`);
    setTimeout(() => { document.getElementById("err").style.display = "none"; }, 6000);
  }

  await setup(viewer, data, hasToken);
}

async function setup(viewer, data, hasTerrain) {
  const { dt, n_steps, vmax, center } = data.meta;
  const cars = data.cars;

  // --- clock: map frame index -> simulated time -------------------------
  const start = C.JulianDate.now();
  const stop = C.JulianDate.addSeconds(start, (n_steps - 1) * dt, new C.JulianDate());
  Object.assign(viewer.clock, {
    startTime: start.clone(), stopTime: stop.clone(), currentTime: start.clone(),
    clockRange: C.ClockRange.LOOP_STOP, multiplier: PLAYBACK, shouldAnimate: true,
  });
  viewer.timeline.zoomTo(start, stop);

  const times = [];
  for (let f = 0; f < n_steps; f++)
    times.push(C.JulianDate.addSeconds(start, f * dt, new C.JulianDate()));

  // --- drape heights onto Cesium terrain (fallback: our DEM z - geoid) --
  const heights = cars.map(c => c.h.map(h => h - GEOID));  // ellipsoidal fallback
  if (hasTerrain) {
    try {
      const carto = [];
      cars.forEach(c => c.lng.forEach((lng, f) =>
        carto.push(C.Cartographic.fromDegrees(lng, c.lat[f]))));
      await C.sampleTerrainMostDetailed(viewer.terrainProvider, carto);
      let k = 0;
      cars.forEach((c, i) => c.lng.forEach((_, f) => {
        const z = carto[k++].height;
        if (Number.isFinite(z)) heights[i][f] = z;
      }));
    } catch (e) { console.warn("terrain sampling failed, using DEM z:", e); }
  }

  // --- one entity per car: position + velocity orientation + speed color -
  const frameAt = (time) => Math.min(n_steps - 1, Math.max(0,
    Math.round(C.JulianDate.secondsDifference(time, start) / dt)));

  cars.forEach((c, i) => {
    const pos = new C.SampledPositionProperty();
    for (let f = 0; f < n_steps; f++) {
      pos.addSample(times[f], C.Cartesian3.fromDegrees(
        c.lng[f], c.lat[f], heights[i][f] + CAR_Z_OFFSET));
    }
    pos.setInterpolationOptions({
      interpolationDegree: 1, interpolationAlgorithm: C.LinearApproximation });

    const color = new C.CallbackProperty(() => {
      const f = frameAt(viewer.clock.currentTime);
      if (c.crash[f]) return C.Color.RED.withAlpha(0.95);
      return speedColor(c.spd[f] / vmax);
    }, false);

    viewer.entities.add({
      position: pos,
      orientation: new C.VelocityOrientationProperty(pos),
      box: {
        dimensions: new C.Cartesian3(CAR.length, CAR.width, CAR.height),
        material: new C.ColorMaterialProperty(color),
        outline: true, outlineColor: C.Color.BLACK.withAlpha(0.4),
      },
    });
  });

  // --- camera: angled view over the cohort ------------------------------
  viewer.camera.flyTo({
    destination: C.Cartesian3.fromDegrees(center[0], center[1] - 0.006, 900),
    orientation: { heading: 0.0, pitch: C.Math.toRadians(-35), roll: 0.0 },
    duration: 0,
  });

  // --- HUD --------------------------------------------------------------
  const s = data.summary || {};
  const hVal = document.getElementById("h-valid");
  hVal.textContent = s.valid_run ? "✓ clean" : "✗ violations";
  hVal.className = "v " + (s.valid_run ? "ok" : "bad");

  viewer.clock.onTick.addEventListener(() => {
    const f = frameAt(viewer.clock.currentTime);
    let moving = 0, crashed = 0;
    for (const c of cars) {
      if (c.crash[f]) crashed++; else if (c.spd[f] > 1.0) moving++;
    }
    document.getElementById("h-clock").textContent = (f * dt).toFixed(1) + "s";
    document.getElementById("h-moving").textContent = moving;
    document.getElementById("h-crash").textContent = crashed;
    document.getElementById("h-trips").textContent = s.trips ?? "—";
  });
}

// red (stopped) -> yellow -> green (fast)
function speedColor(frac) {
  const f = Math.max(0, Math.min(1, frac));
  let r, g, b;
  if (f < 0.5) { const t = f / 0.5; r = 230; g = 60 + 150 * t; b = 50; }
  else { const t = (f - 0.5) / 0.5; r = 230 - 180 * t; g = 210 - 30 * t; b = 60 + 20 * t; }
  return C.Color.fromBytes(r | 0, g | 0, b | 0, 235);
}

boot();
