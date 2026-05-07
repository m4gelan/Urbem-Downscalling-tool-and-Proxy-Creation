const API = "";

let map = null;
let domainLayer = null;
let resultsLayer = null;
let intermediateLayer = null;

function valueToColor(t) {
  const r = Math.round(173 + 82 * t);
  const g = Math.round(216 - 216 * t);
  const b = Math.round(230 - 230 * t);
  return "rgb(" + Math.max(0, Math.min(255, r)) + "," + Math.max(0, Math.min(255, g)) + "," + Math.max(0, Math.min(255, b)) + ")";
}

function percentileColor(val, p25, p75, p90, p100) {
  if (val <= 0) return "rgb(230, 245, 255)";
  if (val <= p25 || p25 >= p100) return "rgb(230, 245, 255)";
  if (val <= p75) {
    const t = (p75 - p25) > 0 ? (val - p25) / (p75 - p25) : 0;
    const r = Math.round(173 - 43 * t);
    const g = Math.round(216 - 86 * t);
    const b = Math.round(255 - 25 * t);
    return "rgb(" + Math.max(0, Math.min(255, r)) + "," + Math.max(0, Math.min(255, g)) + "," + Math.max(0, Math.min(255, b)) + ")";
  }
  if (val <= p90) {
    const t = (p90 - p75) > 0 ? (val - p75) / (p90 - p75) : 0;
    const r = Math.round(130 + 103 * t);
    const g = Math.round(130 - 100 * t);
    const b = Math.round(230 - 131 * t);
    return "rgb(" + Math.max(0, Math.min(255, r)) + "," + Math.max(0, Math.min(255, g)) + "," + Math.max(0, Math.min(255, b)) + ")";
  }
  const t = (p100 - p90) > 0 ? (val - p90) / (p100 - p90) : 1;
  const r = Math.round(233 + 22 * t);
  const g = Math.round(30 - 30 * t);
  const b = Math.round(99 - 99 * t);
  return "rgb(" + Math.min(255, Math.max(0, r)) + "," + Math.max(0, Math.min(255, g)) + "," + Math.max(0, Math.min(255, b)) + ")";
}

function formatLegendValue(logVal) {
  const val = Math.pow(10, logVal);
  if (val >= 1000) return val.toExponential(0);
  if (val >= 1) return val.toFixed(2);
  if (val >= 0.01) return val.toFixed(4);
  return val.toExponential(1);
}

function updateValueLegend(vminLog, vmaxLog, show) {
  const el = document.getElementById("value-legend");
  if (!el) return;
  if (!show) {
    el.classList.add("hidden");
    return;
  }
  el.classList.remove("hidden");
  el.querySelector(".value-legend-max").textContent = formatLegendValue(vmaxLog);
  el.querySelector(".value-legend-min").textContent = formatLegendValue(vminLog);
}
let currentConfig = null;
let configPath = null;
let configDir = null;
let lastOutputFolder = null;
let lastOutputPath = null;
let lastSourceType = null;
let intermediatesList = null;
let resultsSnapsData = null;
let workflowStep = 1;
let proxiesPhaseComplete = false;
let proxyBuildStartMs = 0;

function resolvedEmissionFolder(config) {
  if (!config) return "";
  if (config.output_folder) return config.output_folder;
  const p = config.paths || {};
  if (p.output_root && p.emission_region) {
    const root = String(p.output_root).replace(/[/\\]+$/, "");
    const reg = String(p.emission_region).replace(/^[/\\]+|[/\\]+$/g, "");
    return root + "/" + "emission" + "/" + reg;
  }
  return "";
}

function initMap() {
  map = L.map("map").setView([39.67, 20.85], 10);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "OpenStreetMap",
  }).addTo(map);
}

function updateDomainOnMap(domainCfg) {
  if (!map || !domainCfg) return;
  if (domainLayer) {
    map.removeLayer(domainLayer);
    domainLayer = null;
  }
  fetch(API + "/api/domain/geojson", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ domain: domainCfg }),
  })
    .then((r) => r.json())
    .then((geojson) => {
      if (geojson.type === "Feature") {
        domainLayer = L.geoJSON(geojson, {
          style: { color: "#3498db", weight: 2, fillOpacity: 0.1 },
        }).addTo(map);
        map.fitBounds(domainLayer.getBounds());
      }
    })
    .catch((err) => console.error("Domain GeoJSON error:", err));
}

function showLayerPanel() {
  const panel = document.getElementById("layer-panel");
  const outDir = lastOutputFolder || resolvedEmissionFolder(currentConfig);
  if (!panel || !outDir) return;
  panel.classList.remove("hidden");
  const viewToggle = document.getElementById("view-toggle");
  if (viewToggle && lastOutputPath) viewToggle.classList.remove("hidden");
  fetchIntermediatesList();
  if (lastSourceType === "area") fetchResultsSnaps();
  buildLayerOptions();
}

function fetchResultsSnaps() {
  if (!lastOutputPath || lastSourceType !== "area") return;
  fetch(API + "/api/output/snaps-and-pollutants", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_path: lastOutputPath }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) return;
      resultsSnapsData = data;
      buildLayerOptions();
    })
    .catch(() => {});
}

function fetchIntermediatesList() {
  const outDir = lastOutputFolder || resolvedEmissionFolder(currentConfig);
  if (!outDir || !currentConfig || !currentConfig.domain) return;
  fetch(API + "/api/intermediates/list", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_folder: outDir }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) return;
      intermediatesList = data;
      buildLayerOptions();
    })
    .catch(() => {});
}

const LAYER_LABELS = {
  domain: "Domain bounds",
  cams_grid: "CAMS reprojection grid",
  proxies: "Proxies",
  cams_emissions: "CAMS emissions (pre-downscaling)",
  downscaled: "Downscaled emissions",
  line_cams: "Line CAMS (pre-downscaling)",
  line_downscaled: "Line downscaled",
  results: "Results",
};

function buildLayerOptions(rebuildSelect) {
  const layerSel = document.getElementById("layer-type");
  const container = document.getElementById("layer-options");
  if (!container || !layerSel) return;
  if (rebuildSelect !== false) {
    const currentValue = layerSel.value;
    const layers = intermediatesList?.layers;
    if (layers && layers.length) {
      layerSel.innerHTML = layers.map((id) =>
        '<option value="' + id + '">' + (LAYER_LABELS[id] || id) + "</option>"
      ).join("");
      if (layers.includes(currentValue)) layerSel.value = currentValue;
    } else {
      layerSel.innerHTML = '<option value="domain">Domain bounds</option>' +
        '<option value="results">Results</option>';
      if (currentValue === "domain" || currentValue === "results") layerSel.value = currentValue;
    }
  }
  const layerType = layerSel.value || "domain";
  container.innerHTML = "";

  if (layerType === "proxies") {
    const div = document.createElement("div");
    div.className = "form-group";
    if (intermediatesList && intermediatesList.proxies && intermediatesList.proxies.length > 0) {
      div.innerHTML = '<label>Proxy</label><select id="layer-proxy">' +
        intermediatesList.proxies.map((p) => '<option value="' + p + '">' + p + "</option>").join("") +
        "</select>";
    } else {
      div.innerHTML = '<span class="layer-msg">No proxies available (area sources only)</span>';
    }
    container.appendChild(div);
  }

  if (layerType === "line_cams" || layerType === "line_downscaled") {
    const pollDiv = document.createElement("div");
    pollDiv.className = "form-group";
    pollDiv.innerHTML = '<label>Pollutant</label><select id="layer-line-pollutant">' +
      (intermediatesList?.pollutants || ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"])
        .map((p) => '<option value="' + p + '">' + p + "</option>").join("") +
      "</select>";
    container.appendChild(pollDiv);
  }

  if (layerType === "cams_emissions") {
    const modeDiv = document.createElement("div");
    modeDiv.className = "form-group";
    modeDiv.innerHTML = '<label>View</label><select id="layer-cams-mode">' +
      '<option value="total">Total (pre-downscaling)</option>' +
      '<option value="sector">By sector (GNFR)</option>' +
      "</select>";
    container.appendChild(modeDiv);
    const sectorDiv = document.createElement("div");
    sectorDiv.className = "form-group";
    sectorDiv.id = "layer-cams-sector-wrap";
    if (intermediatesList && intermediatesList.cams_sectors && intermediatesList.cams_sectors.length > 0) {
      sectorDiv.innerHTML = '<label>Sector (GNFR)</label><select id="layer-sector">' +
        intermediatesList.cams_sectors.map((s) => '<option value="' + s + '">' + s.replace(/_/g, " ") + "</option>").join("") +
        "</select>";
    }
    container.appendChild(sectorDiv);
    const pollDiv = document.createElement("div");
    pollDiv.className = "form-group";
    pollDiv.innerHTML = '<label>Pollutant</label><select id="layer-pollutant">' +
      (intermediatesList?.pollutants || ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"])
        .map((p) => '<option value="' + p + '">' + p + "</option>").join("") +
      "</select>";
    container.appendChild(pollDiv);
    const camsModeSel = document.getElementById("layer-cams-mode");
    const camsSectorWrap = document.getElementById("layer-cams-sector-wrap");
    function toggleCamsSector() {
      if (camsSectorWrap) camsSectorWrap.classList.toggle("hidden", camsModeSel?.value !== "sector");
    }
    toggleCamsSector();
    camsModeSel?.addEventListener("change", toggleCamsSector);
  }

  if (layerType === "downscaled" && intermediatesList) {
    const modeDiv = document.createElement("div");
    modeDiv.className = "form-group";
    modeDiv.innerHTML = '<label>View</label><select id="layer-downscale-mode">' +
      '<option value="total">Total (downscaled)</option>' +
      '<option value="sector">By sector (GNFR)</option>' +
      '<option value="snap">By SNAP</option>' +
      "</select>";
    container.appendChild(modeDiv);
    const sectorDiv = document.createElement("div");
    sectorDiv.className = "form-group";
    sectorDiv.id = "layer-downscale-sector-wrap";
    if (intermediatesList.cams_sectors && intermediatesList.cams_sectors.length > 0) {
      sectorDiv.innerHTML = '<label>Sector</label><select id="layer-downscale-sector">' +
        intermediatesList.cams_sectors.map((s) => '<option value="' + s + '">' + s.replace(/_/g, " ") + "</option>").join("") +
        "</select>";
    }
    container.appendChild(sectorDiv);
    const snapDiv = document.createElement("div");
    snapDiv.className = "form-group hidden";
    snapDiv.id = "layer-downscale-snap-wrap";
    if (intermediatesList.snap_ids && intermediatesList.snap_ids.length > 0) {
      snapDiv.innerHTML = '<label>SNAP</label><select id="layer-downscale-snap">' +
        intermediatesList.snap_ids.map((id) => '<option value="' + id + '">' + id + "</option>").join("") +
        "</select>";
    }
    container.appendChild(snapDiv);
    const pollDiv = document.createElement("div");
    pollDiv.className = "form-group";
    pollDiv.innerHTML = '<label>Pollutant</label><select id="layer-downscale-pollutant">' +
      (intermediatesList.pollutants || ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"])
        .map((p) => '<option value="' + p + '">' + p + "</option>").join("") +
      "</select>";
    container.appendChild(pollDiv);
    const modeSel = document.getElementById("layer-downscale-mode");
    const sectorWrap = document.getElementById("layer-downscale-sector-wrap");
    const snapWrap = document.getElementById("layer-downscale-snap-wrap");
    function toggleDownscaleOptions() {
      const m = modeSel?.value || "total";
      if (sectorWrap) sectorWrap.classList.toggle("hidden", m !== "sector");
      if (snapWrap) snapWrap.classList.toggle("hidden", m !== "snap");
    }
    toggleDownscaleOptions();
    modeSel?.addEventListener("change", toggleDownscaleOptions);
  }

  if (layerType === "results" && lastSourceType === "line") {
    const pollDiv = document.createElement("div");
    pollDiv.className = "form-group";
    pollDiv.innerHTML = '<label>Pollutant</label><select id="layer-results-pollutant">' +
      ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]
        .map((p) => '<option value="' + p + '">' + p + "</option>").join("") +
      "</select>";
    container.appendChild(pollDiv);
    document.getElementById("layer-results-pollutant")?.addEventListener("change", () => applySelectedLayer());
  }

  if (layerType === "results" && lastSourceType === "area") {
    const snapByPoll = resultsSnapsData?.snap_by_pollutant || {};
    const hasData = Object.keys(snapByPoll).filter((p) => snapByPoll[p] && snapByPoll[p].length > 0);
    const pollutants = hasData.length > 0
      ? hasData.sort()
      : (resultsSnapsData?.pollutants || ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]);
    const pollDiv = document.createElement("div");
    pollDiv.className = "form-group";
    pollDiv.innerHTML = '<label>Pollutant</label><select id="layer-results-pollutant">' +
      pollutants.map((p) => '<option value="' + p + '">' + p + "</option>").join("") +
      "</select>";
    container.appendChild(pollDiv);
    const modeDiv = document.createElement("div");
    modeDiv.className = "form-group";
    modeDiv.innerHTML = '<label>View</label><select id="layer-results-mode">' +
      '<option value="total">Total</option>' +
      '<option value="snap">By SNAP</option>' +
      "</select>";
    container.appendChild(modeDiv);
    const snapDiv = document.createElement("div");
    snapDiv.className = "form-group hidden";
    snapDiv.id = "layer-results-snap-wrap";
    container.appendChild(snapDiv);
    function updateResultsSnapOptions() {
      const poll = document.getElementById("layer-results-pollutant")?.value || "NOx";
      const snaps = resultsSnapsData?.snap_by_pollutant?.[poll] || [];
      snapDiv.innerHTML = snaps.length ? '<label>SNAP</label><select id="layer-results-snap">' +
        snaps.map((id) => '<option value="' + id + '">' + id + "</option>").join("") +
        "</select>" : '<span class="layer-msg">No SNAP with data for ' + poll + "</span>";
      snapDiv.classList.toggle("hidden", document.getElementById("layer-results-mode")?.value !== "snap");
    }
    updateResultsSnapOptions();
    document.getElementById("layer-results-pollutant")?.addEventListener("change", updateResultsSnapOptions);
    document.getElementById("layer-results-mode")?.addEventListener("change", function () {
      snapDiv.classList.toggle("hidden", this.value !== "snap");
    });
  }
}

function getLayerParams() {
  const layerType = document.getElementById("layer-type")?.value || "domain";
  const params = {
    output_folder: lastOutputFolder || resolvedEmissionFolder(currentConfig),
    layer_type: layerType,
    domain: currentConfig && currentConfig.domain,
    output_path: lastOutputPath,
    source_type: lastSourceType || "area",
  };
  if (layerType === "proxies") {
    const sel = document.getElementById("layer-proxy");
    params.proxy_name = sel ? sel.value : null;
  }
  if (layerType === "line_cams" || layerType === "line_downscaled") {
    params.pollutant = document.getElementById("layer-line-pollutant")?.value || "NOx";
  }
  if (layerType === "cams_emissions") {
    params.mode = document.getElementById("layer-cams-mode")?.value || "total";
    params.sector = document.getElementById("layer-sector")?.value || null;
    params.pollutant = document.getElementById("layer-pollutant")?.value || "NOx";
  }
  if (layerType === "downscaled") {
    const modeSel = document.getElementById("layer-downscale-mode");
    params.mode = modeSel ? modeSel.value : "sector";
    params.pollutant = (document.getElementById("layer-downscale-pollutant")?.value) || "NOx";
    if (params.mode === "sector") {
      params.sector = document.getElementById("layer-downscale-sector")?.value || null;
    } else {
      params.snap_id = parseInt(document.getElementById("layer-downscale-snap")?.value || "0", 10);
    }
  }
  if (layerType === "results" && lastSourceType === "area") {
    params.pollutant = document.getElementById("layer-results-pollutant")?.value || "NOx";
    params.mode = document.getElementById("layer-results-mode")?.value || "total";
    if (params.mode === "snap") {
      params.snap_id = parseInt(document.getElementById("layer-results-snap")?.value || "0", 10);
    }
  }
  if (layerType === "results" && lastSourceType === "line") {
    params.pollutant = document.getElementById("layer-results-pollutant")?.value || "NOx";
  }
  return params;
}

function applySelectedLayer() {
  const params = getLayerParams();
  if (!params.domain || !params.output_folder) return;
  if (params.layer_type === "proxies" && !params.proxy_name) return;
  if ((params.layer_type === "line_cams" || params.layer_type === "line_downscaled") && !params.pollutant) return;
  if (params.layer_type === "cams_emissions" && params.mode !== "total" && !params.sector) return;
  if (params.layer_type === "downscaled") {
    if (params.mode === "sector" && !params.sector) return;
    if (params.mode === "snap" && params.snap_id == null) return;
  }
  if (params.layer_type === "results" && !params.output_path) return;

  if (intermediateLayer) {
    map.removeLayer(intermediateLayer);
    intermediateLayer = null;
  }
  if (resultsLayer && params.layer_type !== "results") {
    map.removeLayer(resultsLayer);
    resultsLayer = null;
  }

  fetch(API + "/api/intermediates/geojson", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })
    .then((r) => r.json())
    .then((geojson) => {
      if (geojson.error) {
        console.error("Layer GeoJSON error:", geojson.error);
        return;
      }
      const hasValue = geojson.features && geojson.features.some((f) => f.properties && f.properties.value != null);
      const hasNox = geojson.features && geojson.features.some((f) => f.properties && f.properties.NOx > 0);
      let vals = [];
      if (hasValue) {
        vals = geojson.features
          .map((f) => f.properties.value)
          .filter((v) => v != null && v > 0);
      } else if (hasNox) {
        vals = geojson.features
          .map((f) => f.properties.NOx)
          .filter((v) => v > 0);
      }
      vals.sort((a, b) => a - b);
      const usePercentile = vals.length >= 4;
      const p25 = usePercentile ? vals[Math.floor(vals.length * 0.25)] : vals[0];
      const p75 = usePercentile ? vals[Math.floor(vals.length * 0.75)] : vals[vals.length - 1];
      const p90 = usePercentile ? vals[Math.floor(vals.length * 0.90)] : vals[vals.length - 1];
      const p100 = vals.length ? vals[vals.length - 1] : 1;
      const vminLog = vals.length ? Math.log10(Math.max(1e-10, vals[0])) : -10;
      const vmaxLog = vals.length ? Math.log10(vals[vals.length - 1]) : 0;

      function styleFn(feature) {
        const val = feature.properties && (feature.properties.value != null ? feature.properties.value : feature.properties.NOx);
        const hasVal = val != null && val > 0;
        let color = "#666";
        if (hasVal) {
          color = usePercentile
            ? percentileColor(val, p25, p75, p90, p100)
            : valueToColor((Math.log10(val) - vminLog) / (vmaxLog - vminLog || 1));
        }
        const isRaster = params.layer_type === "proxies" || params.layer_type === "cams_emissions" || params.layer_type === "downscaled" || params.layer_type === "line_cams" || params.layer_type === "line_downscaled";
        const isLine = params.source_type === "line";
        const fillOp = hasVal ? (isRaster ? 0.4 : (params.layer_type === "results" && params.source_type === "area" ? 0.35 : 0.8)) : 0;
        const strokeOp = hasVal ? 0.8 : 0;
        return {
          color: color,
          weight: isLine ? 2 : 3,
          opacity: strokeOp,
          fillOpacity: fillOp,
        };
      }

      if (params.layer_type === "domain") {
        updateValueLegend(0, 0, false);
        if (domainLayer) map.removeLayer(domainLayer);
        domainLayer = L.geoJSON(geojson, {
          style: { color: "#3498db", weight: 2, fillOpacity: 0.1 },
        }).addTo(map);
        map.fitBounds(domainLayer.getBounds());
      } else if (params.layer_type === "cams_grid") {
        updateValueLegend(0, 0, false);
        intermediateLayer = L.geoJSON(geojson, {
          style: { color: "#95a5a6", weight: 1, fillOpacity: 0.02 },
        }).addTo(map);
      } else if (params.layer_type === "results" && (params.source_type === "line" || params.source_type === "point")) {
        updateValueLegend(vminLog, vmaxLog, hasValue || hasNox);
        resultsLayer = L.geoJSON(geojson, {
          style: styleFn,
          pointToLayer: function (feature, latlng) {
            const s = styleFn(feature);
            return L.circleMarker(latlng, {
              radius: 4,
              fillColor: s.color,
              color: "#333",
              weight: 1,
              fillOpacity: s.fillOpacity,
              opacity: s.opacity,
            });
          },
        }).addTo(map);
      } else {
        const legendMin = usePercentile ? Math.log10(Math.max(1e-10, p25)) : vminLog;
        const legendMax = vmaxLog;
        updateValueLegend(legendMin, legendMax, hasValue || hasNox);
        intermediateLayer = L.geoJSON(geojson, {
          style: styleFn,
          pointToLayer: params.source_type === "point" ? function (feature, latlng) {
            const s = styleFn(feature);
            return L.circleMarker(latlng, {
              radius: 4,
              fillColor: s.color,
              color: "#333",
              weight: 1,
              fillOpacity: s.fillOpacity,
              opacity: s.opacity,
            });
          } : undefined,
        }).addTo(map);
      }

      const layer = intermediateLayer || resultsLayer;
      if (layer && layer.getBounds) {
        const bounds = layer.getBounds();
        if (bounds.isValid()) {
          map.fitBounds(bounds, { padding: [20, 20] });
        }
      }
    })
    .catch((err) => console.error("Layer GeoJSON error:", err));
}

function addResultsToMap(outputPath, sourceType, domainCfg) {
  if (!map || !outputPath || !domainCfg) return;
  if (resultsLayer) {
    map.removeLayer(resultsLayer);
    resultsLayer = null;
  }
  fetch(API + "/api/output/geojson", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      output_path: outputPath,
      source_type: sourceType,
      domain: domainCfg,
    }),
  })
    .then((r) => r.json())
    .then((geojson) => {
      if (geojson.error) {
        console.error("Output GeoJSON error:", geojson.error);
        return;
      }
      if (!geojson.features || geojson.features.length === 0) {
        return;
      }
      const hasNox = geojson.features.some((f) => f.properties && f.properties.NOx > 0);
      let noxVals = [];
      if (hasNox) {
        noxVals = geojson.features
          .map((f) => f.properties.NOx)
          .filter((v) => v > 0);
      }
      const vmin = noxVals.length ? Math.log10(Math.max(1e-10, Math.min(...noxVals))) : -10;
      const vmax = noxVals.length ? Math.log10(Math.max(...noxVals)) : 0;
      updateValueLegend(vmin, vmax, noxVals.length > 0);

      function styleFn(feature) {
        const nox = feature.properties && feature.properties.NOx;
        let color = "#666";
        if (nox > 0) {
          const logNox = Math.log10(nox);
          const t = (logNox - vmin) / (vmax - vmin || 1);
          color = valueToColor(Math.max(0, Math.min(1, t)));
        }
        return {
          color: color,
          weight: sourceType === "line" ? 2 : 3,
          opacity: 0.8,
          fillOpacity: sourceType === "area" ? 0.3 : 0,
        };
      }

      resultsLayer = L.geoJSON(geojson, {
        style: styleFn,
        pointToLayer: function (feature, latlng) {
          return L.circleMarker(latlng, {
            radius: 4,
            fillColor: styleFn(feature).color,
            color: "#333",
            weight: 1,
            fillOpacity: 0.8,
          });
        },
      }).addTo(map);

      const bounds = resultsLayer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [20, 20] });
      }
    })
    .catch((err) => console.error("Output GeoJSON error:", err));
}

function loadConfigToForm(config) {
  currentConfig = config;
  document.getElementById("region").value = config.region || "";
  document.getElementById("year").value = config.year || 2019;
  document.getElementById("source-type").value = config.source_type || "area";
  document.getElementById("output-folder").value = config.output_folder || "";
  const d = config.domain || {};
  document.getElementById("nrow").value = d.nrow || 30;
  document.getElementById("ncol").value = d.ncol || 30;
  document.getElementById("xmin").value = d.xmin || 0;
  document.getElementById("ymin").value = d.ymin || 0;
  document.getElementById("xmax").value = d.xmax || 0;
  document.getElementById("ymax").value = d.ymax || 0;
  document.getElementById("crs").value = d.crs || "EPSG:32634";
  const p = config.paths || {};
  document.getElementById("input-root").value = p.input_root || p.data_root || "Input";
  document.getElementById("output-root").value = p.output_root || "Output";
  document.getElementById("proxy-country").value = p.proxy_country || "default";
  document.getElementById("emission-region").value = p.emission_region || config.region || "";
  document.getElementById("cams-folder").value = p.cams_folder || "";
  document.getElementById("proxies-folder").value = p.proxies_folder || "";
  document.getElementById("config-form").classList.remove("hidden");
  updateDomainOnMap(config.domain);
  updatePipelineStages(config.source_type || "area");
  proxiesPhaseComplete = false;
  const cont = document.getElementById("btn-continue-emissions");
  if (cont) cont.disabled = true;
  const viz = document.getElementById("proxy-viz-section");
  if (viz) viz.classList.add("hidden");
  const img = document.getElementById("proxy-preview-img");
  if (img) {
    if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
    img._blobUrl = null;
    img.removeAttribute("src");
  }
  document.getElementById("btn-run").disabled = true;
  setWorkflowStep(1);
}

function formToConfig() {
  const inputRoot = document.getElementById("input-root").value.trim() || "Input";
  const outRoot = document.getElementById("output-root").value.trim();
  const proxyCountry = document.getElementById("proxy-country").value.trim();
  const emissionRegion = document.getElementById("emission-region").value.trim();
  const proxiesOverride = document.getElementById("proxies-folder").value.trim();
  const outputOverride = document.getElementById("output-folder").value.trim();
  const paths = {
    input_root: inputRoot,
    cams_folder: document.getElementById("cams-folder").value.trim(),
  };
  if (outRoot) paths.output_root = outRoot;
  if (proxyCountry) paths.proxy_country = proxyCountry;
  if (emissionRegion) paths.emission_region = emissionRegion;
  if (proxiesOverride) paths.proxies_folder = proxiesOverride;
  const cfg = {
    region: document.getElementById("region").value,
    year: parseInt(document.getElementById("year").value, 10),
    source_type: document.getElementById("source-type").value,
    domain: {
      nrow: parseInt(document.getElementById("nrow").value, 10),
      ncol: parseInt(document.getElementById("ncol").value, 10),
      xmin: parseFloat(document.getElementById("xmin").value),
      ymin: parseFloat(document.getElementById("ymin").value),
      xmax: parseFloat(document.getElementById("xmax").value),
      ymax: parseFloat(document.getElementById("ymax").value),
      crs: document.getElementById("crs").value,
    },
    paths,
  };
  if (outputOverride) cfg.output_folder = outputOverride;
  return cfg;
}

function updatePipelineStages(sourceType) {
  const stages = {
    area: ["Domain loaded", "CAMS grid reprojected", "Proxies loaded", "Emissions downscaled", "Export complete"],
    point: ["Domain loaded", "CAMS point/area loaded", "Points assigned", "Emissions distributed", "Export complete"],
    line: ["Domain loaded", "CAMS reprojected", "Population proxy", "OSM roads fetched", "Emissions to lines", "Export complete"],
  };
  const stageIds = {
    area: ["domain", "cams", "proxies", "downscale", "export"],
    point: ["domain", "cams", "points", "distribute", "export"],
    line: ["domain", "cams", "proxies", "osm", "lines", "export"],
  };
  window._pipelineStageIds = stageIds[sourceType] || stageIds.area;
  const labels = stages[sourceType] || stages.area;
  const ul = document.getElementById("pipeline-stages");
  ul.innerHTML = labels.map((l) =>
    '<li class="stage-item">' +
    '<span class="stage-label">' + l + '</span>' +
    '<div class="stage-progress-wrap"><div class="stage-progress-bar"></div></div>' +
    '</li>'
  ).join("");
}

function initProxyBuildStages() {
  const ul = document.getElementById("proxy-build-stages");
  if (!ul) return;
  const labels = [
    "CORINE grid & warp",
    "CORINE class proxies",
    "E-PRTR / ancillary rasters",
    "Shipping & population",
    "Publish GeoTIFFs",
  ];
  ul.innerHTML = labels.map((l) =>
    '<li class="stage-item">' +
    '<span class="stage-label">' + l + '</span>' +
    '<div class="stage-progress-wrap"><div class="stage-progress-bar"></div></div>' +
    '</li>'
  ).join("");
}

function renderProxyPublishLog(lines) {
  const logEl = document.getElementById("proxy-publish-log");
  const cap = document.getElementById("proxy-publish-log-title");
  if (!logEl) return;
  logEl.innerHTML = "";
  (lines || []).forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    logEl.appendChild(li);
  });
  const has = lines && lines.length > 0;
  logEl.classList.toggle("hidden", !has);
  if (cap) cap.classList.toggle("hidden", !has);
}

function resetProxyBuildStages() {
  const ul = document.getElementById("proxy-build-stages");
  const title = document.getElementById("proxy-build-stages-title");
  if (title) title.classList.add("hidden");
  renderProxyPublishLog([]);
  if (!ul) return;
  ul.classList.add("hidden");
  ul.querySelectorAll(".stage-item").forEach((li) => {
    li.classList.remove("active", "done");
    const bar = li.querySelector(".stage-progress-bar");
    if (bar) {
      bar.style.width = "0%";
      bar.classList.remove("indeterminate");
    }
  });
}

function setStageProgressForList(listRoot, index, allDone) {
  if (!listRoot) return;
  const items = listRoot.querySelectorAll(".stage-item");
  items.forEach((li, i) => {
    const bar = li.querySelector(".stage-progress-bar");
    if (!bar) return;
    li.classList.remove("active", "done");
    if (allDone || i < index) {
      li.classList.add("done");
      bar.style.width = "100%";
      bar.classList.remove("indeterminate");
    } else if (i === index) {
      li.classList.add("active");
      bar.style.width = "";
      bar.classList.add("indeterminate");
    } else {
      bar.style.width = "0%";
      bar.classList.remove("indeterminate");
    }
  });
}

function setStageProgress(index, allDone) {
  setStageProgressForList(document.getElementById("pipeline-stages"), index, allDone);
}

function setProxyBuildStageProgress(index, allDone) {
  setStageProgressForList(document.getElementById("proxy-build-stages"), index, allDone);
}

function showProxyBuildProgressStarted() {
  proxyBuildStartMs = Date.now();
  renderProxyPublishLog([]);
  const title = document.getElementById("proxy-build-stages-title");
  const ul = document.getElementById("proxy-build-stages");
  if (title) title.classList.remove("hidden");
  if (ul) {
    ul.classList.remove("hidden");
    ul.querySelectorAll(".stage-item").forEach((li) => {
      li.classList.remove("active", "done");
      const bar = li.querySelector(".stage-progress-bar");
      if (bar) {
        bar.style.width = "0%";
        bar.classList.remove("indeterminate");
      }
    });
  }
  setProxyBuildStageProgress(0, false);
}

function resumeProxyBuildProgressUi() {
  const title = document.getElementById("proxy-build-stages-title");
  const ul = document.getElementById("proxy-build-stages");
  if (title) title.classList.remove("hidden");
  if (ul) ul.classList.remove("hidden");
  const btnBuild = document.getElementById("btn-build-proxies");
  if (btnBuild) btnBuild.disabled = true;
}

function syncConfigSelectFromPath() {
  const sel = document.getElementById("config-file-select");
  if (!sel) return;
  if (!configPath) {
    sel.value = "";
    return;
  }
  const norm = (p) => String(p).replace(/\\/g, "/").toLowerCase();
  const want = norm(configPath);
  let found = false;
  for (let i = 0; i < sel.options.length; i++) {
    if (norm(sel.options[i].value) === want) {
      sel.selectedIndex = i;
      found = true;
      break;
    }
  }
  if (!found) sel.value = "";
}

function refreshConfigFileDropdown() {
  const sel = document.getElementById("config-file-select");
  if (!sel) return;
  fetch(API + "/api/config/run-files")
    .then((r) => r.json())
    .then((data) => {
      const files = data.files || [];
      sel.innerHTML = '<option value="">Select a file…</option>';
      files.forEach((f) => {
        const opt = document.createElement("option");
        opt.value = f.path;
        opt.textContent = f.name;
        sel.appendChild(opt);
      });
      syncConfigSelectFromPath();
    })
    .catch(() => {});
}

function pollStatus() {
  fetch(API + "/api/status")
    .then((r) => r.json())
    .then((s) => {
      if (s.completed) {
        if (statusInterval) {
          clearInterval(statusInterval);
          statusInterval = null;
        }
        setStageProgress(0, true);
        document.getElementById("btn-run").disabled = false;
        if (s.error) {
          alert("Error: " + s.error);
          const statusEl = document.getElementById("proxies-status");
          if (statusEl) {
            statusEl.classList.remove("hidden");
            statusEl.className = "proxies-status error";
            statusEl.textContent = "Error: " + s.error;
          }
        } else if (s.output_path) {
          const sourceType = s.source_type || (currentConfig && currentConfig.source_type) || "area";
          const domainCfg = currentConfig && currentConfig.domain;
          lastOutputFolder = s.output_folder || (s.output_path ? String(s.output_path).replace(/[/\\][^/\\]*$/, "") : null) || resolvedEmissionFolder(currentConfig);
          lastOutputPath = s.output_path;
          lastSourceType = sourceType;
          showLayerPanel();
          const layerSel = document.getElementById("layer-type");
          if (layerSel) layerSel.value = "results";
          buildLayerOptions();
          applySelectedLayer();
          const statusEl = document.getElementById("proxies-status");
          if (statusEl) {
            statusEl.classList.remove("hidden");
            statusEl.className = "proxies-status ok";
            statusEl.textContent = "Done. Output: " + s.output_path;
          }
        }
        return;
      }
      if (s.stage) {
        const ids = window._pipelineStageIds || ["domain", "cams", "proxies", "downscale", "export"];
        const idx = ids.indexOf(s.stage);
        if (idx >= 0) {
          setStageProgress(idx, false);
        } else if (s.stage === "export") {
          setStageProgress(ids.length - 1, false);
        }
      }
    })
    .catch(() => {});
}

let statusInterval = null;

function setWorkflowStep(n) {
  workflowStep = n;
  const p1 = document.getElementById("panel-step1");
  const p2 = document.getElementById("panel-step2");
  const t1 = document.getElementById("tab-step1");
  const t2 = document.getElementById("tab-step2");
  if (!p1 || !p2 || !t1 || !t2) return;
  if (n === 1) {
    p1.classList.remove("hidden");
    p2.classList.add("hidden");
    t1.classList.add("active");
    t2.classList.remove("active");
    t1.setAttribute("aria-selected", "true");
    t2.setAttribute("aria-selected", "false");
  } else {
    p1.classList.add("hidden");
    p2.classList.remove("hidden");
    t1.classList.remove("active");
    t2.classList.add("active");
    t1.setAttribute("aria-selected", "false");
    t2.setAttribute("aria-selected", "true");
  }
}

function syncProxyModeBlocks() {
  const build = document.getElementById("proxy-mode-build")?.checked;
  const ex = document.getElementById("block-proxy-existing");
  const bd = document.getElementById("block-proxy-build");
  if (ex) ex.classList.toggle("hidden", !!build);
  if (bd) bd.classList.toggle("hidden", !build);
}

function showProxyVizPanel() {
  const viz = document.getElementById("proxy-viz-section");
  if (viz) viz.classList.remove("hidden");
}

async function loadProxyTifList() {
  if (!configDir) return;
  const sel = document.getElementById("proxy-tif-select");
  if (!sel) return;
  const r = await fetch(API + "/api/proxies/tifs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_config: formToConfig(), config_dir: configDir }),
  });
  const data = await r.json();
  sel.innerHTML = "";
  const tifs = data.tifs || [];
  if (tifs.length === 0) {
    sel.appendChild(new Option("(no .tif in folder)", ""));
    return;
  }
  tifs.forEach((t) => sel.appendChild(new Option(t.name, t.name)));
}

async function refreshProxyPreview() {
  const sel = document.getElementById("proxy-tif-select");
  const img = document.getElementById("proxy-preview-img");
  if (!sel || !img || !configDir) return;
  const fn = sel.value;
  if (!fn) {
    if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
    img._blobUrl = null;
    img.removeAttribute("src");
    return;
  }
  const r = await fetch(API + "/api/proxies/preview.png", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_config: formToConfig(),
      config_dir: configDir,
      filename: fn,
      max_side: 640,
    }),
  });
  const ct = r.headers.get("content-type") || "";
  if (!r.ok || !ct.includes("image")) {
    img.alt = "Preview failed";
    if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
    img._blobUrl = null;
    img.removeAttribute("src");
    return;
  }
  const blob = await r.blob();
  if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
  img._blobUrl = URL.createObjectURL(blob);
  img.src = img._blobUrl;
  img.alt = fn;
}

function pollProxyBuildStatus() {
  const btnBuild = document.getElementById("btn-build-proxies");
  fetch(API + "/api/proxies/build-status")
    .then((r) => r.json())
    .then((s) => {
      const el = document.getElementById("proxy-build-status");
      const title = document.getElementById("proxy-build-stages-title");
      const ul = document.getElementById("proxy-build-stages");
      if (!el) return;
      el.classList.remove("hidden");
      const publishLines = s.publish_log || [];
      renderProxyPublishLog(publishLines);
      if (s.running) {
        if (proxyBuildStartMs === 0) proxyBuildStartMs = Date.now();
        if (title) title.classList.remove("hidden");
        if (ul) ul.classList.remove("hidden");
        const n = ul ? ul.querySelectorAll(".stage-item").length : 0;
        if (n > 0) {
          if (publishLines.length > 0) {
            setProxyBuildStageProgress(n - 1, false);
          } else {
            const elapsed = Date.now() - proxyBuildStartMs;
            const idx = Math.min(Math.floor(elapsed / 28000), n - 1);
            setProxyBuildStageProgress(idx, false);
          }
        }
        el.className = "proxies-status";
        el.textContent = s.message || "Building proxies (this can take a long time)...";
        if (btnBuild) btnBuild.disabled = true;
        setTimeout(pollProxyBuildStatus, 1500);
        return;
      }
      proxyBuildStartMs = 0;
      if (btnBuild) btnBuild.disabled = false;
      if (s.error) {
        resetProxyBuildStages();
        el.className = "proxies-status error";
        el.textContent = s.error;
        return;
      }
      if (ul && ul.querySelectorAll(".stage-item").length) {
        if (title) title.classList.remove("hidden");
        ul.classList.remove("hidden");
        setProxyBuildStageProgress(0, true);
      }
      el.className = "proxies-status ok";
      el.textContent = s.message || "Done.";
      if (s.exit_code === 0) {
        proxiesPhaseComplete = true;
        const cont = document.getElementById("btn-continue-emissions");
        if (cont) cont.disabled = false;
        showProxyVizPanel();
        loadProxyTifList().then(() => refreshProxyPreview());
      }
    })
    .catch(() => {
      proxyBuildStartMs = 0;
      if (btnBuild) btnBuild.disabled = false;
    });
}

function startRun() {
  if (!configPath) {
    alert("Save config first or load an existing config.");
    return;
  }
  if (workflowStep !== 2) {
    setWorkflowStep(2);
  }
  document.getElementById("btn-run").disabled = true;
  currentConfig = formToConfig();
  if (resultsLayer) {
    map.removeLayer(resultsLayer);
    resultsLayer = null;
  }
  if (intermediateLayer) {
    map.removeLayer(intermediateLayer);
    intermediateLayer = null;
  }
  const viewToggle = document.getElementById("view-toggle");
  if (viewToggle) viewToggle.classList.add("hidden");
  const statsPanel = document.getElementById("stats-panel");
  if (statsPanel) {
    statsPanel.classList.add("hidden");
    statsPanel.innerHTML = "";
  }
  const mapEl = document.getElementById("map");
  if (mapEl) mapEl.classList.remove("hidden");
  updateValueLegend(0, 0, false);
  setStageProgress(0, false);
  const statusEl = document.getElementById("proxies-status");
  if (statusEl) {
    statusEl.classList.remove("hidden");
    statusEl.className = "proxies-status";
    statusEl.textContent = "Running...";
  }
  fetch(API + "/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_path: configPath }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        alert("Error: " + data.error);
        document.getElementById("btn-run").disabled = false;
        return;
      }
      statusInterval = setInterval(pollStatus, 500);
    })
    .catch((err) => {
      alert("Error: " + err);
      document.getElementById("btn-run").disabled = false;
    });
}

function validateProxies() {
  const config = formToConfig();
  if (!configDir) {
    alert("Load or save config first to set config directory.");
    return;
  }
  fetch(API + "/api/proxies/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_config: config, config_dir: configDir }),
  })
    .then((r) => r.json())
    .then((result) => {
      const el = document.getElementById("proxies-status");
      el.classList.remove("hidden");
      if (result.all_ok) {
        el.className = "proxies-status ok";
        el.textContent = "All proxy files found (" + result.available.length + " files).";
        proxiesPhaseComplete = true;
        const cont = document.getElementById("btn-continue-emissions");
        if (cont) cont.disabled = false;
        showProxyVizPanel();
        loadProxyTifList().then(() => refreshProxyPreview());
      } else {
        el.className = "proxies-status error";
        const missing = result.missing.map((m) => m[0] + ": " + m[1]).join(", ");
        el.textContent = "Missing: " + missing;
        proxiesPhaseComplete = false;
        const cont = document.getElementById("btn-continue-emissions");
        if (cont) cont.disabled = true;
      }
    })
    .catch((err) => {
      document.getElementById("proxies-status").textContent = "Error: " + err;
      document.getElementById("proxies-status").classList.remove("hidden");
    });
}

let statsCharts = [];

function formatStatVal(v) {
  if (v >= 1000) return v.toExponential(2);
  if (v >= 1) return v.toFixed(2);
  if (v >= 0.01) return v.toFixed(4);
  return v.toExponential(2);
}

function loadStatsPanel() {
  const panel = document.getElementById("stats-panel");
  const mapEl = document.getElementById("map");
  if (!panel || !mapEl || !lastOutputPath || !lastOutputFolder) return;
  if (typeof Chart === "undefined") {
    panel.innerHTML = "<p class=\"layer-msg\">Chart.js not loaded. Check network.</p>";
    panel.classList.remove("hidden");
    mapEl.classList.add("hidden");
    return;
  }
  panel.innerHTML = "<p>Loading statistics...</p>";
  panel.classList.remove("hidden");
  mapEl.classList.add("hidden");
  statsCharts.forEach((c) => c.destroy());
  statsCharts = [];

  fetch(API + "/api/output/statistics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      output_path: lastOutputPath,
      output_folder: lastOutputFolder,
      source_type: lastSourceType || "area",
      config_dir: configDir,
    }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        panel.innerHTML = "<p class=\"layer-msg\">Error: " + data.error + "</p>";
        return;
      }
      renderStats(panel, data);
    })
    .catch((err) => {
      panel.innerHTML = "<p class=\"layer-msg\">Error: " + err + "</p>";
    });
}

function renderStats(panel, data) {
  panel.innerHTML = "";
  const poll = data.pollutants || [];

  const summarySection = document.createElement("div");
  summarySection.className = "stats-section";
  summarySection.innerHTML = "<h3>Summary by pollutant</h3>";
  const cardsDiv = document.createElement("div");
  cardsDiv.className = "stats-cards";
  for (const p of poll) {
    const s = data.summary?.[p] || {};
    const card = document.createElement("div");
    card.className = "stats-card";
    card.innerHTML =
      "<div class=\"label\">" + p + "</div>" +
      "<div class=\"value\">Total: " + formatStatVal(s.total || 0) + " " + (data.units || "g/s") + "</div>" +
      "<div style=\"font-size:11px;color:#666;margin-top:4px\">Mean " + formatStatVal(s.mean || 0) + " | Median " + formatStatVal(s.median || 0) + " | Sparsity " + (s.sparsity || 0).toFixed(1) + "%</div>" +
      "<div style=\"font-size:11px;color:#666\">Top 10% cells: " + ((s.top10_share || 0) * 100).toFixed(1) + "% of emissions</div>";
    cardsDiv.appendChild(card);
  }
  summarySection.appendChild(cardsDiv);
  panel.appendChild(summarySection);

  const comps = (data.cams_comparisons && data.cams_comparisons.length > 0)
    ? data.cams_comparisons
    : (data.cams_comparison ? [data.cams_comparison] : []);
  comps.forEach(function (comp) {
    const compSection = document.createElement("div");
    compSection.className = "stats-section";
    const label = comp.label || "Reference";
    compSection.innerHTML = "<h3>Sanity check</h3><p style=\"font-size:12px;color:#666;margin-bottom:8px\">" + label + ". Ratio near 1 = mass conserved.</p>";
    panel.appendChild(compSection);

    const refData = comp.reference || comp.cams;
    const ratios = comp.ratio || comp.pollutant.map(function (_, i) {
      return refData[i] > 0 ? comp.output[i] / refData[i] : 0;
    });
    const allVals = comp.output.concat(refData).filter(function (v) { return v > 0; });
    const logMin = allVals.length ? Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, allVals))) - 0.5) : 0.1;

    const chartWrap = document.createElement("div");
    chartWrap.className = "stats-chart-wrap";
    compSection.appendChild(chartWrap);
    const canvas = document.createElement("canvas");
    chartWrap.appendChild(canvas);
    const c = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: comp.pollutant,
        datasets: [
          { label: "Output", data: comp.output, backgroundColor: "rgba(52,152,219,0.7)" },
          { label: "Reference", data: refData, backgroundColor: "rgba(149,165,166,0.7)" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            type: "logarithmic",
            min: logMin,
          },
        },
      },
    });
    statsCharts.push(c);

    const ratioTitle = document.createElement("p");
    ratioTitle.style.cssText = "font-size:12px;color:#666;margin:12px 0 4px 0";
    ratioTitle.textContent = "Ratio (Output / Reference). Target: 1.0 = mass conserved.";
    compSection.appendChild(ratioTitle);
    const ratioWrap = document.createElement("div");
    ratioWrap.className = "stats-chart-wrap";
    ratioWrap.style.height = "180px";
    compSection.appendChild(ratioWrap);
    const ratioCanvas = document.createElement("canvas");
    ratioWrap.appendChild(ratioCanvas);
    const validRatios = ratios.filter(function (r) { return isFinite(r) && r > 0; });
    const maxRatio = validRatios.length ? Math.max(1, Math.max.apply(null, validRatios)) : 1;
    const ratioC = new Chart(ratioCanvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: comp.pollutant,
        datasets: [{
          label: "Output / Reference",
          data: ratios.map(function (r) { return isFinite(r) && r > 0 ? r : null; }),
          backgroundColor: ratios.map(function (r) {
            return r >= 0.9 && r <= 1.1 ? "rgba(39,174,96,0.7)" : "rgba(231,76,60,0.7)";
          }),
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: maxRatio < 1 ? Math.max(0.1, maxRatio * 1.5) : Math.max(1.2, maxRatio * 1.1),
            ticks: { callback: function (v) { return Number(v).toFixed(2); } },
          },
        },
      },
    });
    statsCharts.push(ratioC);
  });

  if (data.stacked_bar && data.stacked_bar.length > 0) {
    const snapColors = {};
    const snaps = [...new Set(data.stacked_bar.map((d) => d.snap))].sort((a, b) => a - b);
    const palette = ["#3498db", "#27ae60", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"];
    snaps.forEach((s, i) => { snapColors[s] = palette[i % palette.length]; });
    const stackedSection = document.createElement("div");
    stackedSection.className = "stats-section";
    stackedSection.innerHTML = "<h3>Emissions by pollutant and SNAP (stacked)</h3>";
    const chartWrap = document.createElement("div");
    chartWrap.className = "stats-chart-wrap";
    stackedSection.appendChild(chartWrap);
    panel.appendChild(stackedSection);
    const pols = [...new Set(data.stacked_bar.map((d) => d.pollutant))];
    const datasets = snaps.map((snap) => ({
      label: "SNAP " + snap,
      data: pols.map((p) => {
        const r = data.stacked_bar.find((x) => x.pollutant === p && x.snap === snap);
        return r ? r.total : 0;
      }),
      backgroundColor: snapColors[snap],
    }));
    const canvas = document.createElement("canvas");
    chartWrap.appendChild(canvas);
    const c = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: { labels: pols, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
        plugins: { legend: { position: "top" } },
      },
    });
    statsCharts.push(c);
  }

  if (data.radar && data.radar.length > 0) {
    const radarSection = document.createElement("div");
    radarSection.className = "stats-section";
    radarSection.innerHTML = "<h3>Sector fingerprints</h3><p style=\"font-size:12px;color:#666;margin-bottom:8px\">Share of each pollutant per SNAP (normalized to 100%).</p>";
    const chartWrap = document.createElement("div");
    chartWrap.className = "stats-chart-wrap";
    radarSection.appendChild(chartWrap);
    panel.appendChild(radarSection);
    const palette = ["#3498db", "#27ae60", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c", "#34495e", "#e67e22"];
    const normalized = data.radar.map((r) => {
      const vals = poll.map((p) => r[p] || 0);
      const sum = vals.reduce((a, b) => a + b, 0);
      return sum > 0 ? vals.map((v) => 100 * v / sum) : vals;
    });
    const datasets = poll.map((p, pi) => ({
      label: p,
      data: data.radar.map((r, i) => normalized[i][pi]),
      backgroundColor: palette[pi % palette.length],
    }));
    const canvas = document.createElement("canvas");
    chartWrap.appendChild(canvas);
    const c = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: data.radar.map((r) => "SNAP " + r.snap),
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { stacked: true }, y: { stacked: true, max: 100, beginAtZero: true } },
        plugins: { legend: { position: "top" } },
      },
    });
    statsCharts.push(c);
  }

  if (data.cdf && Object.keys(data.cdf).length > 0) {
    const cdfSection = document.createElement("div");
    cdfSection.className = "stats-section";
    cdfSection.innerHTML = "<h3>Emission CDF</h3><p style=\"font-size:12px;color:#666;margin-bottom:8px\">Cumulative % of emissions vs % of cells (ranked by value). Steep = concentrated.</p>";
    const palette = ["#2980b9", "#27ae60", "#c0392b", "#8e44ad", "#d35400", "#16a085", "#2c3e50"];
    poll.forEach((p, i) => {
      const d = data.cdf[p];
      if (!d || !d.pct_cells || d.pct_cells.length < 2) return;
      const step = Math.max(1, Math.floor(d.pct_cells.length / 80));
      const idx = [];
      for (let j = 0; j < d.pct_cells.length; j += step) idx.push(j);
      if (idx[idx.length - 1] !== d.pct_cells.length - 1) idx.push(d.pct_cells.length - 1);
      const labels = idx.map((j) => d.pct_cells[j].toFixed(0) + "%");
      const vals = idx.map((j) => d.pct_emission[j]);
      const chartWrap = document.createElement("div");
      chartWrap.className = "stats-chart-wrap";
      chartWrap.style.height = "200px";
      const subDiv = document.createElement("div");
      subDiv.style.marginBottom = "16px";
      subDiv.innerHTML = "<strong>" + p + "</strong>";
      subDiv.appendChild(chartWrap);
      cdfSection.appendChild(subDiv);
      const canvas = document.createElement("canvas");
      chartWrap.appendChild(canvas);
      const c = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
          labels,
          datasets: [{
            label: "% emissions",
            data: vals,
            borderColor: palette[i % palette.length],
            backgroundColor: palette[i % palette.length] + "30",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: "% of cells" }, min: 0, max: 100 },
            y: { title: { display: true, text: "% of emissions" }, min: 0, max: 100 },
          },
          plugins: { legend: { display: false } },
        },
      });
      statsCharts.push(c);
    });
    panel.appendChild(cdfSection);
  }

  if (data.moran && Object.keys(data.moran).length > 0) {
    const moranSection = document.createElement("div");
    moranSection.className = "stats-section";
    moranSection.innerHTML = "<h3>Spatial autocorrelation (Moran's I)</h3><p style=\"font-size:12px;color:#666;margin-bottom:8px\">Rows = SNAP, cols = pollutant. Color: low = dispersed, high = clustered.</p>";
    const hasSnap = Object.keys(data.moran).some((k) => k.includes("snap"));
    const pollCols = poll.filter((p) => Object.keys(data.moran).some((k) => k.startsWith(p + "_") || k === p));
    const snapRows = hasSnap
      ? [...new Set(Object.keys(data.moran).map((k) => {
          const m = k.match(/snap(\d+)/);
          return m ? parseInt(m[1], 10) : null;
        }).filter(Boolean))].sort((a, b) => a - b)
      : [null];
    const allVals = Object.values(data.moran).filter((v) => v != null);
    const minV = allVals.length ? Math.min(...allVals) : -1;
    const maxV = allVals.length ? Math.max(...allVals) : 1;
    const toColor = (v) => {
      if (v == null) return "#eee";
      const t = maxV > minV ? (v - minV) / (maxV - minV) : 0.5;
      const r = Math.round(52 + (233 - 52) * t);
      const g = Math.round(152 - 152 * t);
      const b = Math.round(219 - 219 * t);
      return "rgb(" + r + "," + g + "," + b + ")";
    };
    const table = document.createElement("table");
    table.className = "stats-moran-table";
    let html = "<tr><th></th>";
    pollCols.forEach((p) => { html += "<th>" + p + "</th>"; });
    html += "</tr>";
    snapRows.forEach((snap) => {
      html += "<tr><th>" + (snap != null ? "SNAP " + snap : "All") + "</th>";
      pollCols.forEach((p) => {
        const key = snap != null ? p + "_snap" + snap : p;
        const v = data.moran[key];
        const col = toColor(v);
        const txt = v != null ? v.toFixed(3) : "-";
        html += "<td style=\"background:" + col + "\">" + txt + "</td>";
      });
      html += "</tr>";
    });
    table.innerHTML = html;
    moranSection.appendChild(table);
    panel.appendChild(moranSection);
  }

  if (data.line_scatter && data.line_scatter.length > 0) {
    const scatterSection = document.createElement("div");
    scatterSection.className = "stats-section";
    scatterSection.innerHTML = "<h3>Segment length vs emission (line sources)</h3><p style=\"font-size:12px;color:#666;margin-bottom:8px\">Expect linear relationship. Outliers may indicate allocation issues.</p>";
    const chartWrap = document.createElement("div");
    chartWrap.className = "stats-chart-wrap";
    scatterSection.appendChild(chartWrap);
    panel.appendChild(scatterSection);
    const byPoll = {};
    data.line_scatter.forEach((d) => {
      if (!byPoll[d.pollutant]) byPoll[d.pollutant] = { x: [], y: [] };
      byPoll[d.pollutant].x.push(d.length_m);
      byPoll[d.pollutant].y.push(d.emission);
    });
    const palette = ["#3498db", "#27ae60", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"];
    const datasets = Object.entries(byPoll).map(([p, d], i) => ({
      label: p,
      data: d.x.map((x, j) => ({ x, y: d.y[j] })),
      backgroundColor: palette[i % palette.length] + "80",
      borderColor: palette[i % palette.length],
      pointRadius: 3,
    }));
    const canvas = document.createElement("canvas");
    chartWrap.appendChild(canvas);
    const c = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: "Length (m)" } }, y: { title: { display: true, text: "Emission (g/s)" } } },
      },
    });
    statsCharts.push(c);
  }

  if (data.line_box && data.line_box.length > 0) {
    const boxSection = document.createElement("div");
    boxSection.className = "stats-section";
    boxSection.innerHTML = "<h3>Emission per meter by road type (line sources)</h3><p style=\"font-size:12px;color:#666;margin-bottom:8px\">Validates weighting scheme.</p>";
    const chartWrap = document.createElement("div");
    chartWrap.className = "stats-chart-wrap";
    boxSection.appendChild(chartWrap);
    panel.appendChild(boxSection);
    const byRoad = {};
    data.line_box.forEach((d) => {
      const key = d.roadtype + " / " + d.pollutant;
      if (!byRoad[key]) byRoad[key] = [];
      byRoad[key] = byRoad[key].concat(d.values);
    });
    const roadTypes = [...new Set(data.line_box.map((d) => d.roadtype))].sort();
    const pols = [...new Set(data.line_box.map((d) => d.pollutant))];
    const labels = roadTypes.flatMap((rt) => pols.map((p) => rt + " / " + p));
    const boxData = labels.map((l) => {
      const vals = byRoad[l] || [];
      if (vals.length === 0) return { min: 0, q1: 0, median: 0, q3: 0, max: 0 };
      const s = vals.slice().sort((a, b) => a - b);
      const q1 = s[Math.floor(s.length * 0.25)];
      const q3 = s[Math.floor(s.length * 0.75)];
      return { min: s[0], q1, median: s[Math.floor(s.length * 0.5)], q3, max: s[s.length - 1] };
    });
    const canvas = document.createElement("canvas");
    chartWrap.appendChild(canvas);
    const c = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Median (g/s per m)",
          data: boxData.map((d) => d.median),
          backgroundColor: "rgba(52,152,219,0.7)",
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        scales: { x: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
    statsCharts.push(c);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initProxyBuildStages();

  fetch(API + "/api/proxies/build-status")
    .then((r) => r.json())
    .then((s) => {
      if (s.running) {
        resumeProxyBuildProgressUi();
        proxyBuildStartMs = 0;
        const el = document.getElementById("proxy-build-status");
        if (el) {
          el.classList.remove("hidden");
          el.className = "proxies-status";
          el.textContent = s.message || "Building proxies...";
        }
        pollProxyBuildStatus();
      }
    })
    .catch(() => {});

  fetch(API + "/api/config/default")
    .then((r) => r.json())
    .then((data) => {
      configDir = data.config_dir;
      configPath = data.config_path;
      loadConfigToForm(data.config);
      refreshConfigFileDropdown();
    })
    .catch(() => {
      loadConfigToForm({
        region: "Ioannina",
        year: 2019,
        source_type: "area",
        domain: { nrow: 30, ncol: 30, xmin: 468812, ymin: 4375636, xmax: 498812, ymax: 4405636, crs: "EPSG:32634" },
        paths: {
          input_root: "Input",
          output_root: "Output",
          proxy_country: "default",
          emission_region: "Ioannina",
          cams_folder: "given_CAMS/CAMS-REG-ANT_v8.1_TNO_ftp/netcdf",
        },
      });
      refreshConfigFileDropdown();
    });

  document.getElementById("config-file-select")?.addEventListener("change", (e) => {
    const path = e.target.value;
    if (!path) return;
    fetch(API + "/api/config/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          alert(data.error);
          syncConfigSelectFromPath();
          return;
        }
        configDir = data.config_dir;
        configPath = data.config_path;
        loadConfigToForm(data.config);
        syncConfigSelectFromPath();
      })
      .catch((err) => {
        alert("Error: " + err);
        syncConfigSelectFromPath();
      });
  });

  document.getElementById("btn-load-config-other")?.addEventListener("click", () => {
    document.getElementById("file-input").click();
  });

  document.getElementById("file-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      fetch(API + "/api/config/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: reader.result }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.error) {
            alert(data.error);
            return;
          }
          configDir = data.config_dir;
          configPath = data.config_path;
          loadConfigToForm(data.config);
          refreshConfigFileDropdown();
        })
        .catch((err) => alert("Error: " + err));
    };
    reader.readAsText(file);
    e.target.value = "";
  });

  document.getElementById("btn-new-config").addEventListener("click", () => {
    loadConfigToForm({
      region: "",
      year: 2019,
      source_type: "area",
      domain: { nrow: 30, ncol: 30, xmin: 0, ymin: 0, xmax: 0, ymax: 0, crs: "EPSG:32634" },
      paths: {
        input_root: "Input",
        output_root: "Output",
        proxy_country: "default",
        emission_region: "",
        cams_folder: "",
      },
    });
    configPath = null;
    syncConfigSelectFromPath();
  });

  document.getElementById("source-type").addEventListener("change", (e) => {
    updatePipelineStages(e.target.value);
  });

  document.getElementById("btn-validate-proxies").addEventListener("click", validateProxies);
  document.getElementById("proxy-mode-existing")?.addEventListener("change", syncProxyModeBlocks);
  document.getElementById("proxy-mode-build")?.addEventListener("change", syncProxyModeBlocks);
  syncProxyModeBlocks();

  fetch(API + "/api/proxies/vector-subsets")
    .then((r) => r.json())
    .then((d) => {
      const sel = document.getElementById("proxy-vector-subset");
      if (!sel) return;
      const subs = d.subsets || [];
      subs.forEach((s) => sel.appendChild(new Option(s, s)));
    })
    .catch(() => {});

  document.getElementById("btn-build-proxies")?.addEventListener("click", async () => {
    if (!configDir) {
      alert("Load or save config first.");
      return;
    }
    const vs = document.getElementById("proxy-vector-subset")?.value || "";
    try {
      const vr = await fetch(API + "/api/proxies/validate-factory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const v = await vr.json();
      if (!v.ok) {
        let msg = v.error || "Factory input validation failed.";
        if (v.missing && v.missing.length) {
          msg += "\n\nMissing or invalid:\n" + v.missing.map((m) => "  " + m.label + "\n    " + m.path).join("\n");
        }
        alert(msg);
        return;
      }
      const r = await fetch(API + "/api/proxies/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vector_subset: vs || null }),
      });
      const data = await r.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      showProxyBuildProgressStarted();
      const el = document.getElementById("proxy-build-status");
      if (el) {
        el.classList.remove("hidden");
        el.className = "proxies-status";
        el.textContent = "Build started...";
      }
      const btnBuild = document.getElementById("btn-build-proxies");
      if (btnBuild) btnBuild.disabled = true;
      pollProxyBuildStatus();
    } catch (err) {
      alert("Error: " + err);
    }
  });

  document.getElementById("config-collapse-toggle")?.addEventListener("click", () => {
    const btn = document.getElementById("config-collapse-toggle");
    const panel = document.getElementById("config-collapse-panel");
    if (!btn || !panel) return;
    const open = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", open ? "false" : "true");
    panel.classList.toggle("hidden", open);
  });

  document.getElementById("btn-proxy-refresh")?.addEventListener("click", () => {
    loadProxyTifList().then(() => refreshProxyPreview());
  });
  document.getElementById("proxy-tif-select")?.addEventListener("change", () => refreshProxyPreview());

  document.getElementById("btn-continue-emissions")?.addEventListener("click", () => {
    if (!proxiesPhaseComplete) return;
    setWorkflowStep(2);
    if (configPath) document.getElementById("btn-run").disabled = false;
    updatePipelineStages(document.getElementById("source-type").value);
  });

  document.getElementById("tab-step1")?.addEventListener("click", () => setWorkflowStep(1));
  document.getElementById("tab-step2")?.addEventListener("click", () => {
    if (!proxiesPhaseComplete) {
      alert("Validate existing proxies or finish a proxy build in step 1 first.");
      return;
    }
    setWorkflowStep(2);
    if (configPath) document.getElementById("btn-run").disabled = false;
  });

  document.getElementById("btn-save-config").addEventListener("click", () => {
    const config = formToConfig();
    const path = configPath || (configDir ? configDir + "/run_config.json" : null);
    if (!path) {
      alert("Set config directory first (load a config).");
      return;
    }
    fetch(API + "/api/config/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, path }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) alert(data.error);
        else {
          configPath = data.path;
          refreshConfigFileDropdown();
          alert("Config saved.");
        }
      })
      .catch((err) => alert("Error: " + err));
  });
  document.getElementById("btn-run").addEventListener("click", startRun);

  document.getElementById("layer-type")?.addEventListener("change", () => buildLayerOptions(false));
  document.getElementById("btn-apply-layer")?.addEventListener("click", () => applySelectedLayer());

  document.getElementById("btn-view-map")?.addEventListener("click", () => {
    const mapEl = document.getElementById("map");
    const statsPanel = document.getElementById("stats-panel");
    const btnMap = document.getElementById("btn-view-map");
    const btnStats = document.getElementById("btn-view-stats");
    if (mapEl) mapEl.classList.remove("hidden");
    if (statsPanel) statsPanel.classList.add("hidden");
    if (btnMap) btnMap.classList.add("active");
    if (btnStats) btnStats.classList.remove("active");
    updateValueLegend(0, 0, false);
  });

  document.getElementById("btn-view-stats")?.addEventListener("click", () => {
    const mapEl = document.getElementById("map");
    const statsPanel = document.getElementById("stats-panel");
    const btnMap = document.getElementById("btn-view-map");
    const btnStats = document.getElementById("btn-view-stats");
    if (mapEl) mapEl.classList.add("hidden");
    if (btnMap) btnMap.classList.remove("active");
    if (btnStats) btnStats.classList.add("active");
    updateValueLegend(0, 0, false);
    loadStatsPanel();
  });
});
