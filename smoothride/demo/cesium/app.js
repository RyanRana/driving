/* SmoothRide Cesium viewer — replays a scene.json (schema v1) of meshed cars on
   3D San Francisco. Terrain + OSM buildings when an ion token is present; flat
   ellipsoid + our extruded GeoJSON buildings otherwise. */

const CFG = window.SMOOTHRIDE_CONFIG || { cesiumIonToken: "" };
const WORLD = "trained";          // which world to animate
const CAR_L = 4.6, CAR_W = 2.0, CAR_H = 1.5;   // meters

async function main() {
  const hasToken = !!CFG.cesiumIonToken;
  if (hasToken) Cesium.Ion.defaultAccessToken = CFG.cesiumIonToken;

  const viewer = new Cesium.Viewer("cesiumContainer", {
    terrainProvider: hasToken
      ? await Cesium.createWorldTerrainAsync()
      : new Cesium.EllipsoidTerrainProvider(),
    animation: true, timeline: true, baseLayerPicker: false, geocoder: false,
  });
  viewer.scene.globe.depthTestAgainstTerrain = true;

  const scene = await (await fetch("public/scene.json")).json();
  if (scene.schema_version !== 1) throw new Error("unsupported schema " + scene.schema_version);
  const meta = scene.meta, world = scene.worlds[WORLD];

  // Buildings: prefer Cesium OSM Buildings (ion); else extrude our GeoJSON.
  if (hasToken) {
    try { viewer.scene.primitives.add(await Cesium.createOsmBuildingsAsync()); }
    catch (e) { console.warn("OSM Buildings failed, using GeoJSON", e); addGeoJsonBuildings(viewer, scene); }
  } else {
    addGeoJsonBuildings(viewer, scene);
  }

  drawRoads(viewer, scene.roads);

  // Time window for playback.
  const start = Cesium.JulianDate.now();
  const stop = Cesium.JulianDate.addSeconds(start, meta.dt * (meta.n_steps - 1), new Cesium.JulianDate());
  viewer.clock.startTime = start.clone();
  viewer.clock.stopTime = stop.clone();
  viewer.clock.currentTime = start.clone();
  viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
  viewer.clock.multiplier = 1.0;
  viewer.timeline.zoomTo(start, stop);

  world.cars.forEach((car) => addCar(viewer, car, start, meta));

  // HUD
  document.getElementById("cars").textContent = world.summary.cars;
  viewer.clock.onTick.addEventListener(() => {
    const frac = Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, start) / (meta.dt);
    const f = Math.max(0, Math.min(world.trips_series.length - 1, Math.round(frac)));
    document.getElementById("trips").textContent = world.trips_series[f];
  });
  document.getElementById("crashed").textContent = world.summary.crashed_end;

  // Frame the city.
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(meta.center[0], meta.center[1], 1800),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(-45), roll: 0 },
    duration: 0,
  });
}

function sampledPosition(car, start, meta) {
  const p = new Cesium.SampledPositionProperty();
  for (let t = 0; t < car.lng.length; t++) {
    const when = Cesium.JulianDate.addSeconds(start, t * meta.dt, new Cesium.JulianDate());
    p.addSample(when, Cesium.Cartesian3.fromDegrees(car.lng[t], car.lat[t], (car.z[t] || 0) + CAR_H / 2));
  }
  return p;
}

function speedColor(spd, vmax) {
  const f = Math.max(0, Math.min(1, spd / (vmax || 16)));
  return Cesium.Color.fromHsl(0.33 * f, 0.9, 0.5);   // red(stopped) -> green(fast)
}

function addCar(viewer, car, start, meta) {
  const pos = sampledPosition(car, start, meta);
  viewer.entities.add({
    position: pos,
    orientation: new Cesium.VelocityOrientationProperty(pos),
    box: {
      dimensions: new Cesium.Cartesian3(CAR_L, CAR_W, CAR_H),
      material: new Cesium.ColorMaterialProperty(new Cesium.CallbackProperty(() => {
        const f = Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, start) / meta.dt;
        const i = Math.max(0, Math.min(car.spd.length - 1, Math.round(f)));
        return car.crash[i] ? Cesium.Color.DIMGRAY : speedColor(car.spd[i], meta.vmax);
      }, false)),
    },
  });
}

function drawRoads(viewer, roads) {
  roads.forEach((seg) => {
    viewer.entities.add({
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(
          [seg[0][0], seg[0][1], seg[0][2], seg[1][0], seg[1][1], seg[1][2]]),
        width: 2, material: Cesium.Color.fromCssColorString("#3a4658"),
        clampToGround: false,
      },
    });
  });
}

function addGeoJsonBuildings(viewer, scene) {
  const fc = scene.buildings;
  if (!fc || !fc.features) return;
  fc.features.forEach((ft) => {
    const ring = ft.geometry.coordinates[0];
    const flat = [];
    ring.forEach((p) => { flat.push(p[0], p[1]); });
    viewer.entities.add({
      polygon: {
        hierarchy: Cesium.Cartesian3.fromDegreesArray(flat),
        extrudedHeight: ft.properties.height || 8,
        material: Cesium.Color.fromCssColorString("#1a2230").withAlpha(0.9),
        outline: false,
      },
    });
  });
}

main().catch((e) => { console.error(e); alert("Viewer error: " + e.message); });
