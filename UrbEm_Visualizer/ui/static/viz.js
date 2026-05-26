/* UrbEm results map — Leaflet layers, layer tree, statistics */
const UrbEmViz = (function () {
  const API = "";
  const ICON_GLYPH = {
    bolt: "⚡",
    factory: "🏭",
    wind: "💨",
    plane: "✈",
    recycle: "♻",
    flame: "🔥",
    droplet: "💧",
    ship: "⚓",
    truck: "🚜",
    leaf: "🌿",
    dot: "●",
  };

  let meta = null;
  let map = null;
  let basemapLayer = null;
  let domainLayer = null;
  let camsLayer = null;
  const areaLayers = {};
  const pointLayers = {};
  let layerState = {};
  let pollutant = "";
  const userThresholds = {};

  const CMAP_GRADIENT = {
    YlOrRd: "linear-gradient(to right, #ffffcc, #ffeda0, #fed976, #feb24c, #fd8d3c, #fc4e2a, #e31a1c, #800026)",
    YlGn: "linear-gradient(to right, #ffffe5, #f7fcb9, #d9f0a3, #addd8e, #78c679, #41ab5d, #238443, #005a32)",
    Oranges: "linear-gradient(to right, #fff5eb, #fee6ce, #fdd0a2, #fdae6b, #fd8d3c, #f16913, #d94801, #8c2d04)",
    PuRd: "linear-gradient(to right, #f7f4f9, #e7e1ef, #d4b9da, #c994c7, #df65b0, #e7298a, #ce1256, #91003f)",
    BuGn: "linear-gradient(to right, #f7fcfd, #e5f5f9, #ccece6, #99d8c9, #66c2a4, #41ae76, #238b45, #005824)",
    Greys: "linear-gradient(to right, #ffffff, #f0f0f0, #d9d9d9, #bdbdbd, #969696, #737373, #525252, #252525)",
    RdPu: "linear-gradient(to right, #fff7f3, #fde0dd, #fcc5c0, #fa9fb5, #f768a1, #dd3497, #ae017e, #7a0177)",
  };

  function activeSectorIds() {
    return Object.keys(layerState).filter((id) => layerState[id].enabled);
  }

  function activeSectorsParam() {
    return activeSectorIds().join(",");
  }

  function activePointSectorsParam() {
    return activeSectorIds().filter((id) => id !== "TOTAL").join(",");
  }

  function sectorMeta(id) {
    return (meta.sectors || []).find((s) => s.id === id) || { id, label: id, accent: "#4f7cff", icon: "dot" };
  }

  function makePointIcon(props) {
    const accents = props.accents || ["#4f7cff"];
    const sectors = props.sectors || [];
    const sm = sectors.length === 1 ? sectorMeta(sectors[0]) : null;
    const glyph = sm ? ICON_GLYPH[sm.icon] || ICON_GLYPH.dot : "◆";
    let bg;
    if (accents.length === 1) {
      bg = accents[0];
    } else {
      const step = 100 / accents.length;
      const stops = accents.map((c, i) => `${c} ${i * step}% ${(i + 1) * step}%`).join(", ");
      bg = `conic-gradient(${stops})`;
    }
    const html =
      `<div class="pin-wrap"><div style="width:32px;height:32px;border-radius:50%;background:${bg};` +
      `display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;box-shadow:0 2px 6px rgba(0,0,0,.45)">` +
      `${glyph}</div></div>`;
    return L.divIcon({ className: "viz-point-icon", html, iconSize: [32, 32], iconAnchor: [16, 16] });
  }

  function scaleForSector(sectorId) {
    if (sectorId === "TOTAL") {
      return (meta.total_scale || {})[pollutant] || {};
    }
    const per = (meta.per_sector_scale || {})[sectorId] || {};
    return per[pollutant] || (meta.sector_scale || {})[pollutant] || {};
  }

  function currentScale() {
    const areaOn = Object.keys(layerState).filter(
      (id) => layerState[id].enabled && layerState[id].areaOn
    );
    if (areaOn.length === 1) return scaleForSector(areaOn[0]);
    return (meta.sector_scale || {})[pollutant] || (meta.total_scale || {})[pollutant] || {};
  }

  function thresholdValue() {
    if (userThresholds[pollutant] != null) return userThresholds[pollutant];
    return (meta.default_thresholds || {})[pollutant] ?? 0;
  }

  function normForThreshold(scale, thr) {
    if (!scale.lower_bound && scale.lower_bound !== 0) return 0;
    const lv = Math.log10(Math.max(thr, 0) + 0.01);
    const denom = scale.upper_bound - scale.lower_bound;
    return Math.min(100, Math.max(0, ((lv - scale.lower_bound) / denom) * 100));
  }

  function fmtSci(v) {
    if (!Number.isFinite(v) || v === 0) return "0";
    const exp = Math.floor(Math.log10(Math.abs(v)));
    const mant = v / 10 ** exp;
    return `${mant.toFixed(1)}×10^${exp}`;
  }

  function setLegend(scale) {
    if (!scale || !scale.legend_ticks) return;
    const bar = document.getElementById("viz-legend");
    bar.style.background = CMAP_GRADIENT[scale.colormap] || CMAP_GRADIENT.YlOrRd;
    const labels = document.getElementById("viz-legend-labels");
    labels.innerHTML = "";
    const ticks = scale.legend_ticks || [];
    const mid = ticks.find((t) => t.role === "mid") || ticks[0];
    const max = ticks.find((t) => t.role === "max") || ticks[ticks.length - 1];
    if (mid) {
      const el = document.createElement("span");
      el.className = "viz-legend-label viz-legend-label--mid";
      el.style.left = Math.min(92, Math.max(8, (mid.norm || 0) * 100)) + "%";
      el.textContent = `median ${mid.label}`;
      labels.appendChild(el);
    }
    if (max) {
      const el = document.createElement("span");
      el.className = "viz-legend-label viz-legend-label--max";
      el.textContent = `max ${max.label}`;
      labels.appendChild(el);
    }
    const thr = thresholdValue();
    const marker = document.getElementById("viz-legend-marker");
    const pct = normForThreshold(scale, thr);
    marker.classList.remove("hidden");
    marker.style.left = pct + "%";
    marker.title = `Hide below ${fmtSci(thr)} kg/yr/cell`;
    const thrLbl = document.getElementById("viz-legend-threshold-label");
    thrLbl.classList.remove("hidden");
    thrLbl.style.left = pct + "%";
    thrLbl.textContent = fmtSci(thr);
  }

  function tileUrl(sectorId) {
    const q = new URLSearchParams({ pollutant, threshold: String(thresholdValue()) });
    if (sectorId === "TOTAL") {
      q.set("sectors", activeSectorIds().filter((id) => id !== "TOTAL").join(","));
    }
    return `${API}/api/viz/tiles/${sectorId}/{z}/{x}/{y}.png?${q}`;
  }

  function redrawAreaLayers() {
    Object.keys(areaLayers).forEach((k) => {
      if (areaLayers[k]) {
        map.removeLayer(areaLayers[k]);
        delete areaLayers[k];
      }
    });
    refreshArea("TOTAL");
    Object.keys(layerState).forEach((sid) => {
      if (sid !== "TOTAL") refreshArea(sid);
    });
  }

  async function applyThreshold() {
    const val = parseFloat(document.getElementById("viz-threshold").value);
    if (!Number.isFinite(val) || val < 0) return;
    userThresholds[pollutant] = val;
    await fetch(API + "/api/viz/threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pollutant, threshold: val }),
    });
    setLegend(scaleForSector(
      Object.keys(layerState).find((id) => layerState[id].enabled && layerState[id].areaOn) || "TOTAL"
    ));
    redrawAreaLayers();
  }

  function areaLayerZIndex(sectorId) {
    const othersOn = Object.keys(layerState).some(
      (id) => id !== "TOTAL" && layerState[id].enabled && layerState[id].areaOn
    );
    if (sectorId === "TOTAL") return othersOn ? 330 : 350;
    return 360;
  }

  function refreshArea(sectorId) {
    const st = layerState[sectorId];
    if (!st || !st.enabled || !st.areaOn) {
      if (areaLayers[sectorId]) {
        map.removeLayer(areaLayers[sectorId]);
        delete areaLayers[sectorId];
      }
      return;
    }
    if (areaLayers[sectorId]) map.removeLayer(areaLayers[sectorId]);
    areaLayers[sectorId] = L.tileLayer(tileUrl(sectorId), {
      maxZoom: 19,
      minZoom: 5,
      opacity: 1,
      zIndex: areaLayerZIndex(sectorId),
    }).addTo(map);
    if (layerState[sectorId].enabled && layerState[sectorId].areaOn) {
      setLegend(scaleForSector(sectorId));
    }
  }

  async function refreshPoints() {
    const q = new URLSearchParams({ pollutant, sectors: activePointSectorsParam() });
    const r = await fetch(`${API}/api/viz/points?${q}`);
    const gj = await r.json();
    Object.keys(pointLayers).forEach((k) => {
      map.removeLayer(pointLayers[k]);
      delete pointLayers[k];
    });
    for (const sid of Object.keys(layerState)) {
      const st = layerState[sid];
      if (!st.enabled || !st.pointOn) continue;
      const feats = gj.features.filter((f) => (f.properties.sectors || []).includes(sid));
      if (!feats.length) continue;
      pointLayers[sid] = L.geoJSON(
        { type: "FeatureCollection", features: feats },
        {
          pointToLayer: (f, latlng) =>
            L.marker(latlng, { icon: makePointIcon(f.properties) }),
          onEachFeature: (f, layer) => {
            layer.on("click", () => {
              const ll = layer.getLatLng();
              openFacility(f.properties, ll);
            });
          },
        }
      ).addTo(map);
    }
  }

  async function openFacility(props, latlng) {
    const q = new URLSearchParams({
      pollutant,
      lon: latlng.lng,
      lat: latlng.lat,
      sectors: activePointSectorsParam(),
    });
    const r = await fetch(`${API}/api/viz/facility?${q}`);
    const data = await r.json();
    if (data.error) {
      console.warn("facility lookup:", data.error);
      return;
    }
    const panel = document.getElementById("facility-panel");
    panel.classList.remove("hidden");

    const sectorLabel = (data.sectors && data.sectors[0] && data.sectors[0].label) || "";
    const title = data.facility_name || sectorLabel || "Point source";
    document.getElementById("facility-title").textContent = title;

    const ds = document.getElementById("facility-dataset");
    ds.classList.remove("hidden", "facility-dataset-warn");
    if (data.dataset) {
      ds.textContent = `Source dataset: ${data.dataset}`;
    } else if (data.match_status === "matched_appointed") {
      ds.textContent = "Source dataset: unknown (re-run proxy matching for metadata)";
      ds.classList.add("facility-dataset-warn");
    } else if (data.match_status) {
      ds.textContent = `Match: ${data.match_status.replace(/_/g, " ")}`;
    } else {
      ds.textContent = "";
      ds.classList.add("hidden");
    }

    const nameEl = document.getElementById("facility-name");
    if (data.facility_name && data.facility_name !== title) {
      nameEl.textContent = data.facility_name;
      nameEl.classList.remove("hidden");
    } else {
      nameEl.textContent = "";
      nameEl.classList.add("hidden");
    }

    const metaDl = document.getElementById("facility-meta");
    metaDl.innerHTML = "";
    (data.details || []).forEach((d) => {
      if (!d.label || d.value == null || d.value === "") return;
      const dt = document.createElement("dt");
      dt.textContent = d.label;
      const dd = document.createElement("dd");
      dd.textContent = d.value;
      metaDl.appendChild(dt);
      metaDl.appendChild(dd);
    });
    if (data.match_distance_km != null) {
      const dt = document.createElement("dt");
      dt.textContent = "Match distance";
      const dd = document.createElement("dd");
      dd.textContent = `${Number(data.match_distance_km).toFixed(2)} km`;
      metaDl.appendChild(dt);
      metaDl.appendChild(dd);
    }

    document.getElementById("facility-coords").textContent =
      `${latlng.lat.toFixed(5)}°, ${latlng.lng.toFixed(5)}°` +
      (data.cams_point_id != null && data.cams_point_id >= 0 ? ` · CAMS #${data.cams_point_id}` : "");

    const tbody = document.getElementById("facility-tbody");
    tbody.innerHTML = "";
    const unit = data.unit || "kg/yr/cell";
    (data.pollutants || []).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${row.pollutant}</td><td>${row.emission_label || fmtSci(row.emission)}</td><td>${unit}</td>`;
      tbody.appendChild(tr);
    });
    if (!tbody.children.length && (data.sectors || []).length) {
      (data.sectors || []).forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td>${pollutant}</td><td>${row.emission_label || fmtSci(row.emission)}</td><td>${unit}</td>`;
        tbody.appendChild(tr);
      });
    }
  }

  async function refreshViewportCard() {
    if (!map) return;
    const b = map.getBounds();
    const r = await fetch(API + "/api/viz/viewport", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pollutant,
        bbox: { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() },
      }),
    });
    const d = await r.json();
    document.getElementById("viewport-emission").textContent =
      `${d.emission_label || "—"} ${d.unit || "kg/yr/cell"}`;
    document.getElementById("viewport-facilities").textContent =
      `${d.facility_count ?? 0} facilities in view`;
    document.getElementById("viewport-dominant").textContent =
      `Dominant: ${d.dominant_sector || "—"}`;
  }

  function closeFacility() {
    document.getElementById("facility-panel").classList.add("hidden");
  }

  async function refreshAllLayers() {
    setLegend(currentScale());
    Object.keys(areaLayers).forEach((k) => {
      if (!layerState[k]?.enabled || !layerState[k]?.areaOn) {
        map.removeLayer(areaLayers[k]);
        delete areaLayers[k];
      }
    });
    refreshArea("TOTAL");
    for (const sid of Object.keys(layerState)) {
      if (sid !== "TOTAL") refreshArea(sid);
    }
    await refreshPoints();
    refreshViewportCard();
  }

  function toggleSector(sectorId, enableOnly) {
    if (!layerState[sectorId]) return;
    if (enableOnly) {
      Object.keys(layerState).forEach((k) => {
        const has = meta.sectors.find((s) => s.id === k);
        layerState[k].enabled = k === sectorId || k === "TOTAL";
        layerState[k].areaOn = layerState[k].enabled && (k === "TOTAL" || has?.has_area);
        layerState[k].pointOn = layerState[k].enabled && !!has?.has_point;
      });
    } else {
      layerState[sectorId].enabled = !layerState[sectorId].enabled;
      const has = meta.sectors.find((s) => s.id === sectorId) || { has_area: false, has_point: false };
      layerState[sectorId].areaOn = layerState[sectorId].enabled && !!has.has_area;
      layerState[sectorId].pointOn = layerState[sectorId].enabled && !!has.has_point;
    }
    buildLayerTree();
    refreshAllLayers();
  }

  function onMapShown() {
    if (!map) return;
    map.invalidateSize();
    redrawAreaLayers();
    refreshViewportCard();
  }

  function showMap() {
    if (typeof window.UrbEmShowScreen === "function") {
      window.UrbEmShowScreen("screen-viz");
    } else {
      document.body.classList.add("viz-mode");
      document.body.classList.remove("analytics-mode");
      document.querySelectorAll(".screen").forEach((el) => el.classList.add("hidden"));
      document.getElementById("screen-viz").classList.remove("hidden");
      onMapShown();
    }
    setTimeout(onMapShown, 120);
  }

  function showAnalytics() {
    if (typeof window.UrbEmShowScreen === "function") {
      window.UrbEmShowScreen("screen-analytics");
    } else {
      document.body.classList.remove("viz-mode");
      document.body.classList.add("analytics-mode");
      document.querySelectorAll(".screen").forEach((el) => el.classList.add("hidden"));
      document.getElementById("screen-analytics").classList.remove("hidden");
    }
    UrbEmStats.open();
  }

  function initLayerState() {
    layerState = {
      TOTAL: { enabled: true, expanded: true, areaOn: true, pointOn: false },
    };
    (meta.sectors || []).forEach((s) => {
      layerState[s.id] = {
        enabled: false,
        expanded: true,
        areaOn: false,
        pointOn: false,
      };
    });
  }

  function buildLayerTree() {
    const tree = document.getElementById("viz-layer-tree");
    tree.innerHTML = "";
    const sectors = [{ id: "TOTAL", label: "TOTAL", accent: "#e8eaf0", has_area: true, has_point: false }];
    (meta.sectors || []).forEach((s) => sectors.push(s));

    sectors.forEach((s) => {
      if (!layerState[s.id]) {
        layerState[s.id] = {
          enabled: s.id === "TOTAL",
          expanded: true,
          areaOn: s.id === "TOTAL" && !!s.has_area,
          pointOn: false,
        };
      }
      const st = layerState[s.id];
      const block = document.createElement("div");
      block.className = "viz-sector";

      const sub = document.createElement("div");
      sub.className = "viz-sublayers" + (st.expanded ? "" : " collapsed");

      const areaChk = document.createElement("input");
      areaChk.type = "checkbox";
      areaChk.checked = st.areaOn;
      areaChk.disabled = !s.has_area;

      const pointChk = document.createElement("input");
      pointChk.type = "checkbox";
      pointChk.checked = st.pointOn;
      pointChk.disabled = !s.has_point;

      const head = document.createElement("div");
      head.className = "viz-sector-head";
      const chk = document.createElement("input");
      chk.type = "checkbox";
      chk.checked = st.enabled;
      chk.addEventListener("change", () => {
        st.enabled = chk.checked;
        st.areaOn = chk.checked && !!s.has_area;
        st.pointOn = chk.checked && !!s.has_point;
        areaChk.checked = st.areaOn;
        pointChk.checked = st.pointOn;
        refreshAllLayers();
      });
      const dot = document.createElement("span");
      dot.className = "viz-sector-dot";
      dot.style.background = s.accent;
      const title = document.createElement("span");
      title.textContent = s.label || s.id;
      title.addEventListener("click", () => {
        st.expanded = !st.expanded;
        sub.classList.toggle("collapsed", !st.expanded);
      });
      head.appendChild(chk);
      head.appendChild(dot);
      head.appendChild(title);
      block.appendChild(head);

      const areaRow = document.createElement("label");
      areaRow.className = "viz-subrow";
      areaChk.addEventListener("change", () => {
        st.areaOn = areaChk.checked;
        refreshAllLayers();
      });
      const areaLbl = document.createElement("span");
      areaLbl.textContent = "Area";
      areaRow.appendChild(areaChk);
      areaRow.appendChild(areaLbl);
      sub.appendChild(areaRow);

      const pointRow = document.createElement("label");
      pointRow.className = "viz-subrow";
      pointChk.addEventListener("change", () => {
        st.pointOn = pointChk.checked;
        refreshPoints();
      });
      const pointLbl = document.createElement("span");
      pointLbl.textContent = "Points";
      pointRow.appendChild(pointChk);
      pointRow.appendChild(pointLbl);
      sub.appendChild(pointRow);

      block.appendChild(sub);
      tree.appendChild(block);
    });
  }

  function initBasemaps() {
    const sel = document.getElementById("viz-basemap");
    sel.innerHTML = "";
    const maps = meta.map_config?.basemaps || {};
    let defaultKey = "dark";
    Object.entries(maps).forEach(([key, m]) => {
      if (m.default) defaultKey = key;
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = m.label || key;
      sel.appendChild(opt);
    });
    sel.value = defaultKey;
    sel.addEventListener("change", () => setBasemap(sel.value));
    setBasemap(defaultKey);
  }

  function setBasemap(key) {
    const m = meta.map_config?.basemaps?.[key];
    if (!m) return;
    if (basemapLayer) map.removeLayer(basemapLayer);
    basemapLayer = L.tileLayer(m.url, { attribution: m.attribution || "", maxZoom: 19 }).addTo(map);
    basemapLayer.bringToBack();
  }

  async function open(outputDir) {
    if (typeof L === "undefined") {
      throw new Error("Leaflet failed to load — hard-refresh the page (Ctrl+F5)");
    }
    const r = await fetch(`${API}/api/viz/open`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_dir: outputDir }),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error + (data.errors ? ": " + data.errors.join("; ") : ""));
    meta = data.meta;
    pollutant = (meta.pollutants || [])[0] || "NOx";
    initLayerState();

    document.body.classList.add("viz-mode");
    document.querySelectorAll(".screen").forEach((el) => el.classList.add("hidden"));
    document.getElementById("screen-viz").classList.remove("hidden");

    const polSel = document.getElementById("viz-pollutant");
    polSel.innerHTML = "";
    (meta.pollutants || []).forEach((p) => {
      const o = document.createElement("option");
      o.value = p;
      o.textContent = p;
      polSel.appendChild(o);
    });
    polSel.value = pollutant;
    polSel.onchange = () => {
      pollutant = polSel.value;
      document.getElementById("viz-threshold").value = thresholdValue();
      redrawAreaLayers();
      refreshPoints();
      refreshViewportCard();
    };
    document.getElementById("viz-threshold").value = thresholdValue();
    document.getElementById("btn-threshold-apply").onclick = applyThreshold;

    if (map) {
      map.remove();
      map = null;
    }
    const b = meta.domain_wgs84;
    map = L.map("viz-map", { zoomControl: false, attributionControl: true }).fitBounds([
      [b.south, b.west],
      [b.north, b.east],
    ]);
    initBasemaps();
    buildLayerTree();

    const dom = await (await fetch(`${API}/api/viz/domain`)).json();
    if (domainLayer) map.removeLayer(domainLayer);
    domainLayer = L.geoJSON(dom, {
      style: { color: "#4f7cff", weight: 2, fillOpacity: 0, dashArray: "6 4" },
    }).addTo(map);

    try {
      const cams = await (await fetch(`${API}/api/viz/cams-grid`)).json();
      if (!cams.error && cams.features) {
        if (camsLayer) map.removeLayer(camsLayer);
        camsLayer = L.geoJSON(cams, {
          style: { fillColor: "#8b91a8", fillOpacity: 0.12, color: "#6b7289", weight: 0.7 },
          interactive: false,
        }).addTo(map);
        domainLayer.bringToFront();
      }
    } catch (e) {
      console.warn("CAMS grid overlay:", e);
    }

    const note = document.getElementById("viz-map-note");
    if (note && meta.domain_crs) {
      note.textContent =
        "Domain outline follows " +
        meta.domain_crs +
        " (4 corners). CAMS cell edges are approximate lat/lon rectangles — lines may not align exactly with emissions.";
    }

    document.getElementById("btn-facility-close").onclick = closeFacility;
    document.getElementById("btn-viz-stats").onclick = showAnalytics;
    document.getElementById("btn-viz-back").onclick = () => close();
    map.on("moveend zoomend", refreshViewportCard);

    await refreshAllLayers();
  }

  function close() {
    document.body.classList.remove("viz-mode");
    document.getElementById("screen-viz").classList.add("hidden");
    document.getElementById("screen-menu-a").classList.remove("hidden");
    closeFacility();
    if (map) {
      map.remove();
      map = null;
    }
  }

  return {
    open,
    close,
    toggleSector,
    showMap,
    showAnalytics,
    onMapShown,
    getPollutant: () => pollutant,
    getThreshold: (pol) => userThresholds[pol || pollutant] ?? (meta?.default_thresholds || {})[pol || pollutant] ?? 0,
  };
})();
