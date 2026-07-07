const API = "";

let activeConfig = null;
let configPath = null;
let expectedSectors = [];
let camsFiles = {};
let defaultEmissionsYear = 2019;
let writerCountry = "";
let lastCheck = null;
let availablePollutants = [];
let runsDir = "";
let outputsDir = "";
let bboxMap = null;
let camsLayer = null;
let drawLayer = null;
let drawnRect = null;
let domainCrs = "EPSG:3035";

function showScreen(id) {
  document.body.classList.toggle("viz-mode", id === "screen-viz");
  document.body.classList.toggle("analytics-mode", id === "screen-analytics");
  document.querySelectorAll(".screen").forEach((el) => el.classList.add("hidden"));
  const screen = document.getElementById(id);
  if (screen) screen.classList.remove("hidden");
  if (id === "screen-viz" && window.UrbEmViz?.onMapShown) {
    window.UrbEmViz.onMapShown();
  }
  const titles = {
    "screen-menu-a": ["UrbEm downscaling tool", "Create, load, or open a downscaling run"],
    "screen-menu-b": ["Configuration", "Review inputs, domain, and output"],
    "screen-writer-b1": ["Configuration writer", "Validate INPUT or set paths manually"],
    "screen-writer-b1b": ["Define inputs filepaths", "CAMS and sector GeoTIFF paths"],
    "screen-writer-b3": ["Bounding box & EPSG", "CAMS grid and target domain"],
    "screen-processing": ["Processing", "Downscaling pipeline"],
    "screen-viz": ["Results", "Emission map"],
    "screen-analytics": ["Analytics", "Sector totals and charts (all sectors)"],
  };
  const t = titles[id];
  if (t) {
    document.getElementById("header-title").textContent = t[0];
    document.getElementById("header-subtitle").textContent = t[1];
  }
}

function setMenuStatus(msg) {
  const el = document.getElementById("menu-a-status");
  if (!el) return;
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.className = "status menu-a-status ok";
  el.textContent = msg;
  el.classList.remove("hidden");
}

async function openVisualization(outputDir) {
  try {
    await UrbEmViz.open(outputDir);
  } catch (e) {
    setMenuStatus("");
    alert(String(e.message || e));
  }
}

async function loadOutputFolder() {
  const errBox = document.getElementById("load-output-errors");
  errBox.classList.add("hidden");
  errBox.innerHTML = "";
  setMenuStatus("");
  const r = await fetch(API + "/api/dialog/pick-folder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "Select downscaling output folder" }),
  });
  const pick = await r.json();
  if (pick.cancelled || !pick.path) return;
  setMenuStatus("Validating output folder…");
  const v = await fetch(API + "/api/viz/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_dir: pick.path }),
  });
  const check = await v.json();
  if (!check.ok) {
    setMenuStatus("");
    errBox.classList.remove("hidden");
    errBox.innerHTML = "<strong>Cannot open folder:</strong><ul>" +
      (check.errors || []).map((e) => "<li>" + e + "</li>").join("") + "</ul>";
    return;
  }
  setMenuStatus("Preparing the visualization map…");
  await openVisualization(pick.path);
  setMenuStatus("");
}

function statusClass(s) {
  if (s === "ok") return "status-ok";
  if (s === "absent" || s === "missing") return "status-" + (s === "absent" ? "absent" : "missing");
  if (s === "n/a") return "status-na";
  if (s === "optional omitted") return "status-omit";
  return "";
}

function statusLabel(s) {
  if (s === "ok") return "OK";
  if (s === "absent") return "Absent";
  if (s === "missing") return "Missing";
  if (s === "n/a") return "N/A";
  if (s === "optional omitted") return "Omitted";
  return s;
}

function renderInputsTable(rows) {
  const body = document.getElementById("inputs-table-body");
  body.innerHTML = "";
  (rows || []).forEach((r) => {
    const tr = document.createElement("tr");
    const opt = r.optional ? " (optional)" : "";
    tr.innerHTML =
      "<td><strong>" +
      r.sector +
      "</strong>" +
      opt +
      "</td>" +
      '<td class="' +
      statusClass(r.point_source) +
      '">' +
      statusLabel(r.point_source) +
      "</td>" +
      '<td class="' +
      statusClass(r.area_weights) +
      '">' +
      statusLabel(r.area_weights) +
      "</td>";
    body.appendChild(tr);
  });
}

function buildPollutantCheckboxes(containerId, selected) {
  const box = document.getElementById(containerId);
  if (!box) return;
  const sel = new Set(selected || []);
  box.innerHTML = "";
  availablePollutants.forEach((p) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = p;
    cb.checked = sel.has(p);
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(p));
    box.appendChild(lab);
  });
}

function collectSelectedPollutants(containerId) {
  const out = [];
  document.querySelectorAll("#" + containerId + ' input[type="checkbox"]:checked').forEach((cb) => {
    out.push(cb.value);
  });
  return out;
}

function selectedEmissionsYear(selectId) {
  const el = document.getElementById(selectId || "emissions-year-select");
  if (!el || !el.value) return defaultEmissionsYear;
  return parseInt(el.value, 10);
}

function syncCamsPathFromYear(year, pathInputId) {
  const rel = camsFiles[String(year)];
  if (rel) document.getElementById(pathInputId || "path-cams").value = rel;
}

function buildEmissionsYearSelect(selectId, selectedYear) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  const years = Object.keys(camsFiles)
    .map((y) => parseInt(y, 10))
    .sort((a, b) => a - b);
  const pick = selectedYear != null ? parseInt(selectedYear, 10) : defaultEmissionsYear;
  sel.innerHTML = "";
  years.forEach((y) => {
    const opt = document.createElement("option");
    opt.value = String(y);
    opt.textContent = String(y);
    if (y === pick) opt.selected = true;
    sel.appendChild(opt);
  });
}

function configLeadText(config) {
  const country = config.country || "";
  const y = config.emissions_year || config.year || defaultEmissionsYear;
  return "Country: " + country + " — CAMS & proxy inventory " + y;
}

function showMenuB(config, tableRows) {
  activeConfig = config;
  showScreen("screen-menu-b");
  document.getElementById("menu-b-lead").textContent = configLeadText(config);
  renderInputsTable(tableRows || []);
  buildPollutantCheckboxes("pollutants-checkboxes", config.pollutants || []);
  updateDomainSummary();
  updateOutputSectionVisibility();
  if (runsDir) document.getElementById("runs-dir-hint").textContent = runsDir;
  const nameEl = document.getElementById("config-name");
  if (configPath) {
    const base = configPath.replace(/\\/g, "/").split("/").pop() || "";
    nameEl.value = base.replace(/\.ya?ml$/i, "");
  } else if (config.country) {
    const y = config.emissions_year || config.year || defaultEmissionsYear;
    nameEl.value = config.country + "_" + y;
  }
  if (config.output) {
    document.querySelector('input[name="out-format"][value="' + config.output.format + '"]').checked = true;
    const pm = config.output.point_matching || {};
    const legacyMode = config.output.layer_mode;
    const procedure = pm.procedure || (legacyMode === "merged" ? "merged" : "separate");
    const unmatched = pm.unmatched || "keep_location";
    document.querySelector('input[name="pm-procedure"][value="' + procedure + '"]').checked = true;
    document.querySelector('input[name="pm-unmatched"][value="' + unmatched + '"]').checked = true;
    syncPointMatchingUi();
    if (config.output.grid_resolution_m != null) {
      const gridEl = document.querySelector(
        'input[name="grid-resolution"][value="' + config.output.grid_resolution_m + '"]'
      );
      if (gridEl) gridEl.checked = true;
    }
    const roadsEl = document.getElementById("roads-export-by-category");
    if (roadsEl) {
      roadsEl.checked = config.output.roads_export === "by_category";
    }
  }
}

function syncPointMatchingUi() {
  const separate = document.querySelector('input[name="pm-procedure"][value="separate"]').checked;
  const nested = document.getElementById("pm-separate-options");
  if (nested) nested.classList.toggle("hidden", !separate);
}

function readPointMatchingConfig() {
  const procedure = document.querySelector('input[name="pm-procedure"]:checked').value;
  const pm = { procedure };
  if (procedure === "separate") {
    pm.unmatched = document.querySelector('input[name="pm-unmatched"]:checked').value;
  }
  return pm;
}

function formatPointMatchStats(stats, sectorLabels) {
  const ids = Object.keys(stats || {});
  if (!ids.length) return "";
  const lines = ids.map((sid) => {
    const s = stats[sid];
    const label = (sectorLabels && sectorLabels[sid]) || sid;
    let line =
      label +
      ": " +
      s.total +
      " CAMS point(s) in domain — " +
      s.facilities_0 +
      " with 0 facility, " +
      s.facilities_1 +
      " with 1, " +
      s.facilities_2plus +
      " with 2+";
    if (s.partial_match > 0) {
      line += ", " + s.partial_match + " partial match";
    }
    return line;
  });
  return lines.join("\n");
}

function partialMatchWarningText(total) {
  if (!total || total < 1) return "";
  const n = total === 1 ? "1 point source is a partial match" : total + " point sources are partial matches";
  return (
    n +
    ": the emission is linked to a facility, but either the CAMS location or the facility " +
    "coordinates fall outside your domain (facility coordinates / CAMS point outside the domain)."
  );
}

function updateDomainSummary() {
  const el = document.getElementById("domain-summary");
  const d = activeConfig && activeConfig.domain;
  if (!d) {
    el.classList.add("hidden");
    return;
  }
  el.classList.remove("hidden");
  el.innerHTML =
    "<strong>Domain</strong> (" +
    d.crs +
    "): X " +
    d.xmin.toFixed(2) +
    " … " +
    d.xmax.toFixed(2) +
    ", Y " +
    d.ymin.toFixed(2) +
    " … " +
    d.ymax.toFixed(2) +
    '<br/><span id="domain-wgs-hint"></span>';
}

function updateOutputSectionVisibility() {
  const outSec = document.getElementById("output-section");
  if (activeConfig && activeConfig.domain) outSec.classList.remove("hidden");
  else outSec.classList.add("hidden");
}

function showMenuASaved(path) {
  document.getElementById("menu-a-saved").classList.remove("hidden");
  document.getElementById("menu-a-saved-path").textContent = path;
  showScreen("screen-menu-a");
}

function roleLabel(role) {
  if (role === "point_source") return "point source";
  if (role === "area_weights") return "area weights";
  return role;
}

function renderMissingList(check) {
  const box = document.getElementById("missing-list");
  const contRow = document.getElementById("check-continue-row");
  const pollBlock = document.getElementById("pollutants-b1-block");
  box.innerHTML = "";
  const missing = check.missing || [];
  const waived = check.accepted_absent || [];

  if (waived.length) {
    const note = document.createElement("p");
    note.className = "lead";
    note.style.fontSize = "0.8rem";
    note.textContent =
      "Accepted as absent: " + waived.map((w) => w.sector + " (" + roleLabel(w.role) + ")").join(", ");
    box.appendChild(note);
  }

  if (check.ok) {
    box.classList.add("hidden");
    contRow.classList.remove("hidden");
    pollBlock.classList.remove("hidden");
    buildPollutantCheckboxes("pollutants-b1-checkboxes", activeConfig?.pollutants || availablePollutants);
    return;
  }

  pollBlock.classList.add("hidden");
  contRow.classList.add("hidden");
  if (!missing.length) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");

  missing.forEach((m) => {
    const row = document.createElement("div");
    row.className = "missing-row" + (m.optional ? " optional-missing" : "");
    if (m.optional && m.prompt) {
      const prompt = document.createElement("div");
      prompt.className = "opt-prompt";
      prompt.textContent = m.prompt;
      row.appendChild(prompt);
    } else {
      const label = document.createElement("span");
      label.className = "fname";
      let text = m.filename || m.path || m.kind;
      if (m.sector) text = m.sector + " — " + text;
      if (m.hint) text += " (" + m.hint + ")";
      label.textContent = text;
      row.appendChild(label);
    }
    if (m.waivable && m.sector && m.kind) {
      const actions = document.createElement("div");
      actions.className = "opt-actions";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "waiver";
      btn.textContent = m.optional ? "Yes, this is normal" : "Is this normal?";
      btn.addEventListener("click", () => markWaiver(m.sector, m.kind));
      actions.appendChild(btn);
      row.appendChild(actions);
    }
    box.appendChild(row);
  });
}

async function markWaiver(sector, role) {
  const country = writerCountry || document.getElementById("country-select").value;
  const r = await fetch(API + "/api/waiver/mark", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sector,
      role,
      country,
      emissions_year: selectedEmissionsYear(),
    }),
  });
  const data = await r.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  if (data.check) updateCheckUi(data.check);
}

function updateCheckUi(check) {
  lastCheck = check;
  const st = document.getElementById("check-status");
  st.classList.remove("hidden");
  st.className = check.ok ? "status ok" : "status error";
  st.textContent = check.message;
  renderMissingList(check);
}

async function runCheckInput() {
  const country = document.getElementById("country-select").value;
  if (!country) {
    alert("Please select a country.");
    return null;
  }
  writerCountry = country;
  document.getElementById("check-status").className = "status info";
  document.getElementById("check-status").textContent = "Checking INPUT…";
  document.getElementById("check-status").classList.remove("hidden");
  document.getElementById("missing-list").innerHTML = "";
  document.getElementById("check-continue-row").classList.add("hidden");
  document.getElementById("pollutants-b1-block").classList.add("hidden");

  const r = await fetch(API + "/api/check-input", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ country, emissions_year: selectedEmissionsYear() }),
  });
  const check = await r.json();
  if (check.error) throw new Error(check.error);
  updateCheckUi(check);
  return check;
}

function updateCrsLabels() {
  const wgs = domainCrs === "EPSG:4326";
  document.getElementById("bb-xmin-l").textContent = wgs ? "Lon min" : "X min (m)";
  document.getElementById("bb-ymin-l").textContent = wgs ? "Lat min" : "Y min (m)";
  document.getElementById("bb-xmax-l").textContent = wgs ? "Lon max" : "X max (m)";
  document.getElementById("bb-ymax-l").textContent = wgs ? "Lat max" : "Y max (m)";
  const note = document.getElementById("b3-map-note");
  if (note) {
    note.textContent =
      "Domain outline follows " +
      domainCrs +
      " (4 corners). CAMS cell edges are approximate lat/lon rectangles — lines may not align exactly.";
  }
}

function readBBoxInputs() {
  return {
    xmin: parseFloat(document.getElementById("bb-xmin").value),
    ymin: parseFloat(document.getElementById("bb-ymin").value),
    xmax: parseFloat(document.getElementById("bb-xmax").value),
    ymax: parseFloat(document.getElementById("bb-ymax").value),
  };
}

function writeBBoxInputs(b) {
  document.getElementById("bb-xmin").value = b.xmin;
  document.getElementById("bb-ymin").value = b.ymin;
  document.getElementById("bb-xmax").value = b.xmax;
  document.getElementById("bb-ymax").value = b.ymax;
}

async function transformBBox(xmin, ymin, xmax, ymax, fromCrs, toCrs) {
  const r = await fetch(API + "/api/domain/transform", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ xmin, ymin, xmax, ymax, from_crs: fromCrs, to_crs: toCrs }),
  });
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data;
}

function setDrawnPolygonFromRing(ring) {
  if (typeof L === "undefined") return;
  if (!bboxMap || !drawLayer || !ring || ring.length < 4) return;
  if (drawnRect) {
    drawLayer.removeLayer(drawnRect);
    drawnRect = null;
  }
  const latlngs = ring.slice(0, -1).map(([lon, lat]) => [lat, lon]);
  drawnRect = L.polygon(latlngs, { color: "#4f7cff", weight: 2, fillOpacity: 0.08 });
  drawLayer.addLayer(drawnRect);
}

async function syncMapFromInputs(fit) {
  const b = readBBoxInputs();
  if ([b.xmin, b.ymin, b.xmax, b.ymax].some((v) => Number.isNaN(v))) return false;
  const wgs = await transformBBox(b.xmin, b.ymin, b.xmax, b.ymax, domainCrs, "EPSG:4326");
  if (wgs.ring) {
    setDrawnPolygonFromRing(wgs.ring);
  } else {
    setDrawnPolygonFromRing([
      [wgs.xmin, wgs.ymin],
      [wgs.xmax, wgs.ymin],
      [wgs.xmax, wgs.ymax],
      [wgs.xmin, wgs.ymax],
      [wgs.xmin, wgs.ymin],
    ]);
  }
  if (fit && drawnRect && bboxMap) {
    bboxMap.fitBounds(drawnRect.getBounds(), { padding: [24, 24] });
  }
  return true;
}

async function showBBoxOnMap() {
  if (!bboxMap) {
    alert("Wait for the map to finish loading.");
    return;
  }
  const b = readBBoxInputs();
  if ([b.xmin, b.ymin, b.xmax, b.ymax].some((v) => Number.isNaN(v))) {
    alert("Enter all four coordinates first.");
    return;
  }
  await syncMapFromInputs(true);
}

async function syncInputsFromBounds(bounds) {
  const t = await transformBBox(
    bounds.getWest(),
    bounds.getSouth(),
    bounds.getEast(),
    bounds.getNorth(),
    "EPSG:4326",
    domainCrs
  );
  writeBBoxInputs(t);
}

function setCamsProgress(pct, msg) {
  const wrap = document.getElementById("cams-progress-wrap");
  wrap.classList.remove("hidden");
  document.getElementById("cams-progress-label").textContent = msg || "Loading…";
  document.getElementById("cams-progress-fill").style.width = Math.round(pct * 100) + "%";
}

async function pollCamsJob(jobId) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const r = await fetch(API + "/api/cams/grid/status/" + jobId);
        const st = await r.json();
        if (st.error && !st.done) {
          reject(new Error(st.error));
          return;
        }
        setCamsProgress(st.progress || 0, st.message || "Loading CAMS grid…");
        if (!st.done) {
          setTimeout(tick, 400);
          return;
        }
        if (st.error) {
          reject(new Error(st.error));
          return;
        }
        resolve(st.geojson);
      } catch (e) {
        reject(e);
      }
    };
    tick();
  });
}

async function initB3Map() {
  if (typeof L === "undefined") {
    throw new Error("Leaflet failed to load — hard-refresh the page (Ctrl+F5)");
  }
  const st = document.getElementById("b3-status");
  const prog = document.getElementById("cams-progress-wrap");
  prog.classList.remove("hidden");
  setCamsProgress(0, "Starting CAMS grid load…");
  st.classList.add("hidden");

  if (bboxMap) {
    bboxMap.remove();
    bboxMap = null;
    camsLayer = null;
    drawLayer = null;
    drawnRect = null;
  }

  const startR = await fetch(API + "/api/cams/grid/start", { method: "POST" });
  const startData = await startR.json();
  if (startData.error) throw new Error(startData.error);

  const gj = await pollCamsJob(startData.job_id);

  bboxMap = L.map("bbox-map", { preferCanvas: true });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "OpenStreetMap",
    maxZoom: 19,
  }).addTo(bboxMap);

  const b = gj.bounds;
  bboxMap.setView([(b.south + b.north) / 2, (b.west + b.east) / 2], 6);

  camsLayer = L.geoJSON(gj, {
    style: { fillColor: "#8b91a8", fillOpacity: 0.22, color: "#6b7289", weight: 0.6 },
    interactive: false,
  }).addTo(bboxMap);

  drawLayer = new L.FeatureGroup();
  bboxMap.addLayer(drawLayer);
  bboxMap.addControl(
    new L.Control.Draw({
      draw: {
        polygon: false,
        polyline: false,
        circle: false,
        marker: false,
        circlemarker: false,
        rectangle: { shapeOptions: { color: "#4f7cff" } },
      },
      edit: { featureGroup: drawLayer, remove: true },
    })
  );
  bboxMap.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [24, 24] });

  bboxMap.on(L.Draw.Event.CREATED, async (e) => {
    if (drawnRect) drawLayer.removeLayer(drawnRect);
    drawnRect = e.layer;
    drawLayer.addLayer(drawnRect);
    await syncInputsFromBounds(drawnRect.getBounds());
  });
  bboxMap.on("draw:edited", async (e) => {
    e.layers.eachLayer(async (layer) => {
      drawnRect = layer;
      await syncInputsFromBounds(layer.getBounds());
    });
  });
  bboxMap.on("draw:deleted", () => {
    drawnRect = null;
  });

  if (activeConfig && activeConfig.domain) {
    const d = activeConfig.domain;
    domainCrs = d.crs || "EPSG:3035";
    document.querySelectorAll(".crs-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.crs === domainCrs);
    });
    updateCrsLabels();
    writeBBoxInputs(d);
    await syncMapFromInputs();
  } else {
    domainCrs = "EPSG:3035";
    updateCrsLabels();
  }

  prog.classList.add("hidden");
  st.classList.remove("hidden");
  const warns = gj.coverage_warnings || [];
  if (warns.length) {
    st.className = "status warn";
    st.innerHTML =
      "CAMS grid: " +
      (gj.cell_count || "?") +
      " cells.<ul class='b3-warn-list'>" +
      warns.map((w) => "<li>" + w + "</li>").join("") +
      "</ul>";
  } else {
    st.className = "status ok";
    st.textContent = "CAMS grid: " + (gj.cell_count || "?") + " cells.";
  }
  setTimeout(() => bboxMap.invalidateSize(), 200);
}

async function openB3() {
  if (!activeConfig || !activeConfig.pollutants || !activeConfig.pollutants.length) {
    alert("Select pollutants in the configuration screen first.");
    return;
  }
  showScreen("screen-writer-b3");
  try {
    await initB3Map();
  } catch (e) {
    document.getElementById("cams-progress-wrap").classList.add("hidden");
    const st = document.getElementById("b3-status");
    st.className = "status error";
    st.textContent = "Error: " + e;
    st.classList.remove("hidden");
  }
}

function pathLineHtml(sector, role, optional) {
  return (
    '<div class="path-line" data-sector="' +
    sector +
    '" data-role="' +
    role +
    '">' +
    '<input type="text" data-role="' +
    role +
    '" readonly placeholder="' +
    roleLabel(role) +
    ' .tif" />' +
    '<button type="button" class="no-file" data-no-file="' +
    sector +
    '" data-role="' +
    role +
    '">No file for this source</button>' +
    '<button type="button" data-pick-sector="' +
    sector +
    '" data-role="' +
    role +
    '">Browse…</button></div>'
  );
}

function buildSectorPathForm() {
  const box = document.getElementById("sector-paths-container");
  box.innerHTML = "";
  expectedSectors.forEach((sec) => {
    const block = document.createElement("div");
    block.className = "sector-path-block";
    block.dataset.sector = sec.id;
    const opt = sec.optional ? " <span style='color:var(--warn)'>optional</span>" : "";
    let html =
      "<h3>" + sec.id + opt + " <span style='color:var(--text-secondary);font-weight:400'>(" + sec.mode + ")</span></h3>";
    if (sec.mode === "both" || sec.mode === "point_only") html += pathLineHtml(sec.id, "point_source", sec.optional);
    if (sec.mode === "both" || sec.mode === "area_only") html += pathLineHtml(sec.id, "area_weights", sec.optional);
    block.innerHTML = html;
    box.appendChild(block);
  });
  buildPollutantCheckboxes("pollutants-b1b-checkboxes", activeConfig?.pollutants || []);
}

function setPathAbsent(line, absent) {
  const inp = line.querySelector("input");
  const noBtn = line.querySelector(".no-file");
  const browse = line.querySelector("[data-pick-sector]");
  if (absent) {
    line.classList.add("absent");
    inp.value = "";
    inp.placeholder = "No file for this source";
    inp.dataset.absent = "1";
    if (noBtn) noBtn.classList.add("active");
    if (browse) browse.disabled = true;
  } else {
    line.classList.remove("absent");
    inp.dataset.absent = "0";
    inp.placeholder = inp.dataset.role === "point_source" ? "point source .tif" : "area weights .tif";
    if (noBtn) noBtn.classList.remove("active");
    if (browse) browse.disabled = false;
  }
}

function fillManualFromConfig(config) {
  const ey = config.emissions_year || defaultEmissionsYear;
  buildEmissionsYearSelect("emissions-year-b1b", ey);
  document.getElementById("path-cams").value = (config.paths || {}).cams || camsFiles[String(ey)] || "";
  const sectors = config.sectors || {};
  document.querySelectorAll(".sector-path-block").forEach((block) => {
    const sid = block.dataset.sector;
    const sec = sectors[sid] || {};
    block.querySelectorAll(".path-line").forEach((line) => {
      const role = line.dataset.role;
      const inp = line.querySelector("input");
      const entry = sec[role];
      if (entry && entry.absent) setPathAbsent(line, true);
      else if (entry && entry.path) {
        setPathAbsent(line, false);
        inp.value = entry.path;
      } else {
        setPathAbsent(line, false);
        inp.value = "";
      }
    });
  });
}

function collectManual() {
  const manual = {
    cams: document.getElementById("path-cams").value.trim(),
    emissions_year: selectedEmissionsYear("emissions-year-b1b"),
    sectors: {},
  };
  document.querySelectorAll(".sector-path-block").forEach((block) => {
    const sid = block.dataset.sector;
    const roles = {};
    block.querySelectorAll(".path-line").forEach((line) => {
      const role = line.dataset.role;
      const inp = line.querySelector("input");
      if (inp.dataset.absent === "1") roles[role] = "__absent__";
      else if (inp.value.trim()) roles[role] = inp.value.trim();
    });
    if (Object.keys(roles).length) manual.sectors[sid] = roles;
  });
  return manual;
}

async function loadExpectedSectors() {
  const r = await fetch(API + "/api/expected-sectors");
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  expectedSectors = data.sectors || [];
  camsFiles = data.cams_files || {};
  defaultEmissionsYear = parseInt(data.default_emissions_year, 10) || 2019;
  buildEmissionsYearSelect("emissions-year-select", defaultEmissionsYear);
  buildEmissionsYearSelect("emissions-year-b1b", defaultEmissionsYear);
}

document.addEventListener("DOMContentLoaded", async () => {
  showScreen("screen-menu-a");
  const pr = await fetch(API + "/api/pollutants");
  const pd = await pr.json();
  availablePollutants = pd.pollutants || [];
  const sr = await fetch(API + "/api/session");
  const sd = await sr.json();
  runsDir = sd.runs_dir || "";
  outputsDir = sd.outputs_dir || "";
  const runsHint = document.getElementById("menu-runs-dir-hint");
  const outputsHint = document.getElementById("menu-outputs-dir-hint");
  if (runsHint && runsDir) runsHint.textContent = runsDir;
  if (outputsHint && outputsDir) outputsHint.textContent = outputsDir;

  fetch(API + "/api/countries")
    .then((r) => r.json())
    .then((data) => {
      const sel = document.getElementById("country-select");
      sel.innerHTML = '<option value="">Select country…</option>';
      (data.countries || []).forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      });
    });

  await loadExpectedSectors();
  buildSectorPathForm();

  document.getElementById("btn-load-config").addEventListener("click", async () => {
    const r = await fetch(API + "/api/dialog/open-yaml", { method: "POST" });
    const data = await r.json();
    if (data.cancelled) return;
    if (data.error) {
      alert(data.error);
      return;
    }
    configPath = data.path;
    showMenuB(data.config, data.inputs_table);
  });

  document.getElementById("btn-create-config").addEventListener("click", async () => {
    activeConfig = null;
    configPath = null;
    writerCountry = "";
    lastCheck = null;
    document.getElementById("menu-a-saved").classList.add("hidden");
    await fetch(API + "/api/waiver/clear", { method: "POST" });
    document.getElementById("check-status").classList.add("hidden");
    document.getElementById("missing-list").classList.add("hidden");
    document.getElementById("check-continue-row").classList.add("hidden");
    document.getElementById("pollutants-b1-block").classList.add("hidden");
    showScreen("screen-writer-b1");
  });

  document.getElementById("btn-check-input").addEventListener("click", () => {
    runCheckInput().catch((e) => {
      document.getElementById("check-status").className = "status error";
      document.getElementById("check-status").textContent = "Error: " + e;
    });
  });

  document.getElementById("btn-check-continue").addEventListener("click", async () => {
    const country = writerCountry || document.getElementById("country-select").value;
    const pollutants = collectSelectedPollutants("pollutants-b1-checkboxes");
    if (!pollutants.length) {
      alert("Select at least one pollutant.");
      return;
    }
    const r = await fetch(API + "/api/writer/from-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        country,
        pollutants,
        emissions_year: selectedEmissionsYear(),
      }),
    });
    const data = await r.json();
    if (!r.ok || !data.config) {
      alert(data.check?.message || data.error || "Check not complete.");
      if (data.check) updateCheckUi(data.check);
      return;
    }
    showMenuB(data.config, data.inputs_table);
  });

  document.getElementById("btn-define-inputs").addEventListener("click", async () => {
    const country = document.getElementById("country-select").value;
    if (!country) {
      alert("Please select a country first.");
      return;
    }
    writerCountry = country;
    const r = await fetch(API + "/api/writer/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        country,
        emissions_year: selectedEmissionsYear("emissions-year-b1b"),
      }),
    });
    const data = await r.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    activeConfig = data.config;
    buildSectorPathForm();
    fillManualFromConfig(activeConfig);
    showScreen("screen-writer-b1b");
  });

  document.getElementById("sector-paths-container").addEventListener("click", async (ev) => {
    const noBtn = ev.target.closest(".no-file");
    if (noBtn) {
      const line = noBtn.closest(".path-line");
      setPathAbsent(line, !line.classList.contains("absent"));
      return;
    }
    const btn = ev.target.closest("[data-pick-sector]");
    if (!btn || btn.disabled) return;
    const r = await fetch(API + "/api/dialog/pick-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: btn.dataset.pickSector + " " + btn.dataset.role }),
    });
    const data = await r.json();
    if (data.cancelled || !data.path) return;
    const line = btn.closest(".path-line");
    setPathAbsent(line, false);
    line.querySelector("input").value = data.path;
  });

  document.getElementById("emissions-year-select").addEventListener("change", () => {
    if (lastCheck) runCheckInput().catch(console.error);
  });

  document.getElementById("emissions-year-b1b").addEventListener("change", () => {
    syncCamsPathFromYear(selectedEmissionsYear("emissions-year-b1b"));
  });

  document.getElementById("screen-writer-b1b").addEventListener("click", async (ev) => {
    const pick = ev.target.closest("[data-pick]");
    if (!pick || pick.dataset.pick !== "cams") return;
    const r = await fetch(API + "/api/dialog/pick-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "CAMS NetCDF" }),
    });
    const data = await r.json();
    if (!data.cancelled && data.path) document.getElementById("path-cams").value = data.path;
  });

  document.getElementById("btn-b1b-apply").addEventListener("click", async () => {
    const manual = collectManual();
    const pollutants = collectSelectedPollutants("pollutants-b1b-checkboxes");
    if (!manual.cams) {
      alert("CAMS path is required.");
      return;
    }
    if (!pollutants.length) {
      alert("Select at least one pollutant.");
      return;
    }
    if (!activeConfig) {
      const nr = await fetch(API + "/api/writer/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          country: writerCountry || document.getElementById("country-select").value,
          emissions_year: selectedEmissionsYear("emissions-year-b1b"),
        }),
      });
      activeConfig = (await nr.json()).config;
    }
    const r = await fetch(API + "/api/writer/apply-manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ manual, pollutants }),
    });
    const data = await r.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    showMenuB(data.config, data.inputs_table);
  });

  document.getElementById("btn-open-b3").addEventListener("click", () => openB3());

  document.querySelectorAll(".crs-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const next = btn.dataset.crs;
      if (next === domainCrs) return;
      const b = readBBoxInputs();
      if (![b.xmin, b.ymin, b.xmax, b.ymax].some((v) => Number.isNaN(v))) {
        writeBBoxInputs(await transformBBox(b.xmin, b.ymin, b.xmax, b.ymax, domainCrs, next));
      }
      domainCrs = next;
      document.querySelectorAll(".crs-btn").forEach((x) => x.classList.toggle("active", x.dataset.crs === domainCrs));
      updateCrsLabels();
      await syncMapFromInputs();
    });
  });

  ["bb-xmin", "bb-ymin", "bb-xmax", "bb-ymax"].forEach((id) => {
    document.getElementById(id).addEventListener("input", () => syncMapFromInputs(false).catch(console.error));
  });

  document.getElementById("btn-b3-show-box").addEventListener("click", () => showBBoxOnMap().catch(console.error));

  document.getElementById("btn-b3-apply").addEventListener("click", async () => {
    const b = readBBoxInputs();
    if ([b.xmin, b.ymin, b.xmax, b.ymax].some((v) => Number.isNaN(v))) {
      alert("Enter a full bounding box.");
      return;
    }
    const r = await fetch(API + "/api/config/domain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        domain: { crs: domainCrs, xmin: b.xmin, ymin: b.ymin, xmax: b.xmax, ymax: b.ymax },
      }),
    });
    const data = await r.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    activeConfig = data.config;
    const hint = document.getElementById("domain-wgs-hint");
    if (data.domain_wgs84 && hint) {
      const w = data.domain_wgs84;
      hint.textContent =
        "WGS84 bounds: lon " +
        w.xmin.toFixed(4) +
        "–" +
        w.xmax.toFixed(4) +
        ", lat " +
        w.ymin.toFixed(4) +
        "–" +
        w.ymax.toFixed(4);
    }
    showMenuB(data.config, data.inputs_table);
  });

  document.getElementById("btn-b3-back").addEventListener("click", async () => {
    const r = await fetch(API + "/api/session");
    const data = await r.json();
    if (data.config) showMenuB(data.config, data.inputs_table);
    else showScreen("screen-menu-b");
  });

  document.getElementById("pollutants-checkboxes").addEventListener("change", async () => {
    const pollutants = collectSelectedPollutants("pollutants-checkboxes");
    if (!pollutants.length || !activeConfig) return;
    const r = await fetch(API + "/api/config/pollutants", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pollutants }),
    });
    const data = await r.json();
    if (data.config) activeConfig = data.config;
  });

  document.querySelectorAll('input[name="pm-procedure"]').forEach((el) => {
    el.addEventListener("change", syncPointMatchingUi);
  });
  syncPointMatchingUi();

  document.getElementById("btn-save-config").addEventListener("click", async () => {
    const name = document.getElementById("config-name").value.trim();
    if (!name) {
      alert("Enter a configuration file name.");
      return;
    }
    const format = document.querySelector('input[name="out-format"]:checked').value;
    const pointMatching = readPointMatchingConfig();
    const gridResolutionM = parseInt(document.querySelector('input[name="grid-resolution"]:checked').value, 10);
    const roadsExport = document.getElementById("roads-export-by-category").checked
      ? "by_category"
      : "aggregated";
    await fetch(API + "/api/config/output", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        output: {
          format,
          point_matching: pointMatching,
          grid_resolution_m: gridResolutionM,
          roads_export: roadsExport,
        },
      }),
    });
    const pollutants = collectSelectedPollutants("pollutants-checkboxes");
    if (pollutants.length) {
      await fetch(API + "/api/config/pollutants", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pollutants }),
      });
    }
    const r = await fetch(API + "/api/config/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await r.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    configPath = data.path;
    showMenuASaved(data.path);
  });

  let downscaleJobId = null;
  let downscalePollTimer = null;

  function sectorProgressPct(s) {
    if (s.status === "done") return 100;
    if (s.status === "waiting") return 0;
    if (s.progress != null) return s.progress;
    return 0;
  }

  function renderProcessingState(state) {
    const list = document.getElementById("processing-sectors");
    const alert = document.getElementById("processing-alert");
    const matchBox = document.getElementById("processing-match-summary");
    list.innerHTML = "";

    if (matchBox) {
      const stats = state.point_match_stats;
      const labels = {};
      (state.sectors || []).forEach((s) => {
        labels[s.id] = s.label || s.id;
      });
      const text = formatPointMatchStats(stats, labels);
      if (text) {
        matchBox.textContent = text;
        matchBox.classList.remove("hidden");
      } else {
        matchBox.textContent = "";
        matchBox.classList.add("hidden");
      }
    }

    const partialBox = document.getElementById("processing-partial-warning");
    if (partialBox) {
      const ptext = partialMatchWarningText(state.partial_match_total);
      if (ptext) {
        partialBox.textContent = ptext;
        partialBox.classList.remove("hidden");
      } else {
        partialBox.textContent = "";
        partialBox.classList.add("hidden");
      }
    }

    (state.sectors || []).forEach((s) => {
      const row = document.createElement("div");
      row.className = "processing-row processing-row--" + (s.status || "waiting");
      const head = document.createElement("div");
      head.className = "processing-row-head";
      const name = document.createElement("div");
      name.className = "sector-name";
      name.textContent = s.label || s.id;
      const st = document.createElement("div");
      st.className = "processing-status " + (s.status || "waiting");
      st.textContent = s.status || "waiting";
      head.appendChild(name);
      head.appendChild(st);
      row.appendChild(head);

      const pct = sectorProgressPct(s);
      const barWrap = document.createElement("div");
      barWrap.className = "processing-bar-wrap";
      const track = document.createElement("div");
      track.className = "progress-track processing-bar-track";
      const fill = document.createElement("div");
      fill.className = "progress-fill processing-bar-fill";
      if (s.status === "running" && pct < 3) {
        track.classList.add("processing-bar-indeterminate");
        fill.style.width = "30%";
      } else {
        fill.style.width = pct + "%";
      }
      if (s.status === "done") fill.classList.add("done");
      if (s.status === "error") fill.classList.add("error");
      track.appendChild(fill);
      barWrap.appendChild(track);

      const cap = document.createElement("div");
      cap.className = "processing-bar-caption";
      if (s.status === "running" && s.step) {
        cap.textContent = s.step;
      } else if (s.status === "done") {
        cap.textContent = "Complete";
      } else if (s.status === "error") {
        cap.textContent = s.step || "Failed";
      } else if (s.status === "waiting") {
        cap.textContent = "Waiting";
      }
      barWrap.appendChild(cap);
      row.appendChild(barWrap);
      list.appendChild(row);
    });

    if (state.error) {
      alert.textContent = state.error;
      alert.classList.remove("hidden");
    } else {
      alert.classList.add("hidden");
      alert.textContent = "";
    }
    const lead = document.getElementById("processing-lead");
    const doneMsg = document.getElementById("processing-done-msg");
    if (state.status === "running") {
      lead.textContent = "Downscaling in progress…";
      if (doneMsg) {
        doneMsg.classList.add("hidden");
        doneMsg.textContent = "";
      }
    } else if (state.status === "done") {
      lead.textContent = "Downscaling finished successfully.";
      if (doneMsg) {
        doneMsg.textContent = state.output_dir
          ? "Output saved to " + state.output_dir + ". Opening the results map…"
          : "All sectors processed. Opening the results map…";
        doneMsg.classList.remove("hidden");
      }
      document.getElementById("btn-processing-close").classList.remove("hidden");
      document.getElementById("btn-processing-cancel").classList.add("hidden");
    } else if (state.status === "error") {
      lead.textContent = "Downscaling stopped due to an error.";
      if (doneMsg) {
        doneMsg.classList.add("hidden");
        doneMsg.textContent = "";
      }
      document.getElementById("btn-processing-close").classList.remove("hidden");
      document.getElementById("btn-processing-cancel").classList.add("hidden");
    } else if (state.status === "cancelled") {
      lead.textContent = "Downscaling cancelled.";
      if (doneMsg) {
        doneMsg.classList.add("hidden");
        doneMsg.textContent = "";
      }
      document.getElementById("btn-processing-close").classList.remove("hidden");
      document.getElementById("btn-processing-cancel").classList.add("hidden");
    }
  }

  async function pollDownscale(jobId) {
    const r = await fetch(API + "/api/downscale/status/" + jobId);
    const data = await r.json();
    if (data.state) renderProcessingState(data.state);
    if (data.done) {
      clearInterval(downscalePollTimer);
      downscalePollTimer = null;
      if (data.state?.status === "done" && data.state?.output_dir) {
        await openVisualization(data.state.output_dir);
      }
    }
  }

  async function runDownscaleJob(partialMatchHandling) {
    document.getElementById("btn-processing-close").classList.add("hidden");
    document.getElementById("btn-processing-cancel").classList.remove("hidden");
    document.getElementById("processing-alert").classList.add("hidden");
    const doneMsg = document.getElementById("processing-done-msg");
    if (doneMsg) {
      doneMsg.classList.add("hidden");
      doneMsg.textContent = "";
    }
    showScreen("screen-processing");
    const body = { config_path: configPath };
    if (partialMatchHandling) body.partial_match_handling = partialMatchHandling;
    const r = await fetch(API + "/api/downscale/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.error) {
      const alert = document.getElementById("processing-alert");
      alert.textContent = data.error;
      alert.classList.remove("hidden");
      return;
    }
    downscaleJobId = data.job_id;
    renderProcessingState({ status: "running", sectors: [] });
    downscalePollTimer = setInterval(() => pollDownscale(downscaleJobId), 1000);
    pollDownscale(downscaleJobId);
  }

  function showPartialMatchModal(preData) {
    const modal = document.getElementById("partial-match-modal");
    const text = document.getElementById("partial-match-modal-text");
    const counts = document.getElementById("partial-match-counts");
    const total = preData.partial_match_total || 0;
    const n =
      total === 1
        ? "1 point source is a partial match"
        : total + " point sources are partial matches";
    text.textContent =
      n +
      ": the emission is linked to a facility, but only one of the CAMS location and the " +
      "facility coordinates falls inside your bounding box.";
    counts.innerHTML = "";
    const fac = preData.facility_outside_domain || 0;
    const cams = preData.cams_outside_domain || 0;
    const li1 = document.createElement("li");
    li1.textContent =
      fac === 1
        ? "1 facility falls outside your domain"
        : fac + " facilities fall outside your domain";
    counts.appendChild(li1);
    const li2 = document.createElement("li");
    li2.textContent =
      cams === 1
        ? "1 CAMS point source falls outside your domain"
        : cams + " CAMS point sources fall outside your domain";
    counts.appendChild(li2);
    modal.classList.remove("hidden");
  }

  function hidePartialMatchModal() {
    document.getElementById("partial-match-modal").classList.add("hidden");
  }

  async function openB3FromSavedConfig() {
    if (!activeConfig && configPath) {
      const r = await fetch(API + "/api/config/load-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config_path: configPath }),
      });
      const data = await r.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      activeConfig = data.config;
    }
    await openB3();
  }

  document.getElementById("btn-partial-adjust").addEventListener("click", async () => {
    hidePartialMatchModal();
    await openB3FromSavedConfig();
  });

  document.getElementById("btn-partial-facility-attrib").addEventListener("click", () => {
    hidePartialMatchModal();
    runDownscaleJob("facility_or_drop");
  });

  document.getElementById("btn-start-downscaling").addEventListener("click", async () => {
    const pre = await fetch(API + "/api/downscale/precheck", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_path: configPath }),
    });
    const preData = await pre.json();
    if (preData.error) {
      alert(preData.error);
      return;
    }
    const partialTotal = preData.partial_match_total || 0;
    if (partialTotal >= 1) {
      showPartialMatchModal(preData);
      return;
    }
    runDownscaleJob(null);
  });

  document.getElementById("btn-processing-cancel").addEventListener("click", async () => {
    if (!downscaleJobId) return;
    await fetch(API + "/api/downscale/cancel/" + downscaleJobId, { method: "POST" });
  });

  document.getElementById("btn-processing-close").addEventListener("click", () => {
    if (downscalePollTimer) clearInterval(downscalePollTimer);
    showMenuASaved(configPath);
  });

  document.getElementById("btn-load-output-folder").addEventListener("click", loadOutputFolder);

  document.getElementById("btn-back-menu-a").addEventListener("click", () => showScreen("screen-menu-a"));
  document.getElementById("btn-b1-back").addEventListener("click", () => showScreen("screen-menu-a"));
  document.getElementById("btn-b1b-back").addEventListener("click", () => showScreen("screen-writer-b1"));

  document.getElementById("btn-analytics-back").addEventListener("click", () => {
    if (window.UrbEmViz?.showMap) UrbEmViz.showMap();
    else showScreen("screen-viz");
  });
});

window.UrbEmShowScreen = showScreen;
