/* Analytics page — totals table, composition, radar; pollutant-specific spatial charts */
const UrbEmStats = (function () {
  const API = "";
  let data = null;
  let pollutant = "";
  let charts = {};
  const radarHidden = new Set();

  function destroyCharts() {
    Object.values(charts).forEach((c) => c && c.destroy());
    charts = {};
  }

  function sectorOrder() {
    const ids = (data.sectors || []).slice();
    return ids;
  }

  function renderSummary() {
    const table = document.getElementById("analytics-summary");
    if (!table) return;
    const pols = data.pollutants || [];
    const bySector = {};
    (data.summary || []).forEach((row) => {
      if (!bySector[row.sector_id]) {
        bySector[row.sector_id] = { label: row.sector, cells: {} };
      }
      bySector[row.sector_id].cells[row.pollutant] = row.total_label;
    });

    const sectorRows = Object.keys(bySector)
      .filter((sid) => sid !== "TOTAL")
      .sort((a, b) => String(bySector[a].label).localeCompare(String(bySector[b].label)));
    if (bySector.TOTAL) sectorRows.push("TOTAL");

    const head =
      "<thead><tr><th>Sector</th>" +
      pols.map((p) => `<th>${p}</th>`).join("") +
      "</tr></thead>";
    const body =
      "<tbody>" +
      sectorRows
        .map((sid) => {
          const row = bySector[sid];
          const cls = sid === "TOTAL" ? ' class="row-total"' : "";
          return (
            `<tr${cls}><td>${row.label}</td>` +
            pols.map((p) => `<td>${row.cells[p] || "—"}</td>`).join("") +
            "</tr>"
          );
        })
        .join("") +
      "</tbody>";
    table.innerHTML = head + body;

    const hint = document.getElementById("summary-table-hint");
    if (hint) {
      hint.textContent =
        `Domain totals (${data.unit || "kg/yr/cell"}) from all area sectors — not filtered by map layer selection.`;
    }
  }

  function renderComposition() {
    const pols = data.pollutants || [];
    const ctx = document.getElementById("chart-composition");
    const wrap = ctx.parentElement;
    wrap.style.minHeight = 48 + pols.length * 36 + "px";
    if (charts.composition) charts.composition.destroy();

    const sectorIds = sectorOrder();
    const datasets = sectorIds.map((sid) => {
      const firstPol = (data.composition || {})[pols[0]] || [];
      const row0 = firstPol.find((s) => s.sector_id === sid);
      return {
        label: row0?.label || sid,
        data: pols.map((p) => {
          const seg = ((data.composition || {})[p] || []).find((s) => s.sector_id === sid);
          return seg ? seg.value : 0;
        }),
        backgroundColor: row0?.color || "#4f7cff",
        sector_id: sid,
      };
    });

    charts.composition = new Chart(ctx, {
      type: "bar",
      data: { labels: pols, datasets },
      options: {
        indexAxis: "y",
        scales: {
          x: {
            stacked: true,
            ticks: {
              color: "#8b91a8",
              callback: (v) => (v >= 1e4 ? v.toExponential(1) : v),
            },
          },
          y: { stacked: true, ticks: { color: "#e8eaf0" } },
        },
        plugins: {
          legend: { labels: { color: "#8b91a8", boxWidth: 12 } },
          tooltip: {
            callbacks: {
              label: (c) => {
                const p = pols[c.dataIndex];
                const seg = ((data.composition || {})[p] || []).find(
                  (s) => s.sector_id === datasets[c.datasetIndex].sector_id
                );
                if (!seg) return c.dataset.label;
                return `${seg.label}: ${seg.value_label} (${seg.percent}% of ${p})`;
              },
            },
          },
        },
        onClick: (_, els) => {
          if (!els.length || !window.UrbEmViz) return;
          const sid = datasets[els[0].datasetIndex]?.sector_id;
          if (sid) UrbEmViz.toggleSector(sid);
        },
      },
    });
  }

  function syncMapThreshold() {
    if (!data?.spatial || !window.UrbEmViz?.getThreshold) return;
    const thr = UrbEmViz.getThreshold(pollutant);
    if (data.spatial[pollutant]) data.spatial[pollutant].threshold = thr;
  }

  function renderRadarToggles() {
    const box = document.getElementById("radar-toggles");
    box.innerHTML = "";
    (data.radar || []).forEach((s) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "radar-toggle" + (radarHidden.has(s.sector_id) ? "" : " on");
      btn.style.borderColor = s.color;
      btn.textContent = s.label;
      btn.onclick = () => {
        if (radarHidden.has(s.sector_id)) radarHidden.delete(s.sector_id);
        else radarHidden.add(s.sector_id);
        btn.classList.toggle("on");
        renderRadar();
      };
      box.appendChild(btn);
    });
  }

  function renderRadar() {
    const ctx = document.getElementById("chart-radar");
    if (charts.radar) charts.radar.destroy();
    const sets = (data.radar || [])
      .filter((s) => !radarHidden.has(s.sector_id))
      .map((s) => ({
        label: s.label,
        data: s.values_pct || s.values,
        borderColor: s.color,
        backgroundColor: s.color + "33",
        pointBackgroundColor: s.color,
        sector_id: s.sector_id,
        raw: s.values,
        raw_labels: s.value_labels,
      }));
    charts.radar = new Chart(ctx, {
      type: "radar",
      data: {
        labels: data.pollutants || [],
        datasets: sets,
      },
      options: {
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: {
              color: "#8b91a8",
              backdropColor: "transparent",
              stepSize: 20,
              callback: (v) => v + "%",
            },
            pointLabels: { color: "#e8eaf0", font: { size: 11 } },
          },
        },
        plugins: {
          legend: { labels: { color: "#8b91a8" } },
          tooltip: {
            callbacks: {
              label: (c) => {
                const raw = sets[c.datasetIndex]?.raw?.[c.dataIndex];
                const lbl = sets[c.datasetIndex]?.raw_labels?.[c.dataIndex];
                return `${c.dataset.label}: ${c.formattedValue}% (${lbl || raw})`;
              },
            },
          },
        },
        onClick: (_, els) => {
          if (!els.length || !window.UrbEmViz) return;
          const ds = charts.radar?.data?.datasets?.[els[0].datasetIndex];
          if (ds?.sector_id) UrbEmViz.toggleSector(ds.sector_id);
        },
      },
    });
  }

  function fmtLogTick(v) {
    if (!Number.isFinite(v) || v <= 0) return "";
    const exp = Math.floor(Math.log10(v));
    const mant = v / 10 ** exp;
    return `${mant.toFixed(1)}e${exp}`;
  }

  function renderHistogram() {
    syncMapThreshold();
    const sp = (data.spatial || {})[pollutant] || {};
    const hist = sp.histogram || {};
    const ctx = document.getElementById("chart-histogram");
    if (charts.histogram) charts.histogram.destroy();
    const thr = sp.threshold || 0;
    const centers = hist.bin_centers || [];
    charts.histogram = new Chart(ctx, {
      type: "bar",
      data: {
        labels: centers,
        datasets: [{ label: "Cells", data: hist.counts || [], backgroundColor: "#c45ab3" }],
      },
      options: {
        scales: {
          x: {
            type: "logarithmic",
            ticks: {
              color: "#8b91a8",
              maxRotation: 0,
              autoSkip: true,
              maxTicksLimit: 8,
              callback: (v) => fmtLogTick(Number(v)),
            },
            title: { display: true, text: "kg/yr/cell", color: "#8b91a8" },
          },
          y: { ticks: { color: "#8b91a8" }, title: { display: true, text: "Cell count", color: "#8b91a8" } },
        },
        plugins: { legend: { display: false } },
      },
      plugins: [{
        id: "thresholdLine",
        afterDraw(chart) {
          if (!thr || !centers.length) return;
          const { ctx: c, chartArea, scales } = chart;
          const x = scales.x.getPixelForValue(thr);
          if (x < chartArea.left || x > chartArea.right) return;
          c.save();
          c.strokeStyle = "#4f7cff";
          c.lineWidth = 2;
          c.setLineDash([6, 4]);
          c.beginPath();
          c.moveTo(x, chartArea.top);
          c.lineTo(x, chartArea.bottom);
          c.stroke();
          c.fillStyle = "#4f7cff";
          c.font = "11px sans-serif";
          c.fillText("map threshold", x + 4, chartArea.top + 12);
          c.restore();
        },
      }],
    });
    const note = document.getElementById("histogram-note");
    if (note) {
      note.textContent =
        `TOTAL layer, ${pollutant}. Vertical line = current map hide-below threshold (${sp.threshold_label || thr}).`;
    }
  }

  function renderLorenz() {
    const sp = (data.spatial || {})[pollutant] || {};
    const l = sp.lorenz || {};
    const xs = l.x_pct || [];
    const ys = l.y_cum_pct || [];
    const pct90 = l.pct90_x;
    const ctx = document.getElementById("chart-lorenz");
    if (charts.lorenz) charts.lorenz.destroy();

    let markIdx = -1;
    for (let i = 0; i < ys.length; i++) {
      if (ys[i] >= 90) {
        markIdx = i;
        break;
      }
    }
    const markData = xs.map((_, i) => (i === markIdx ? ys[i] : null));

    charts.lorenz = new Chart(ctx, {
      type: "line",
      data: {
        labels: xs,
        datasets: [
          {
            label: "Cumulative emission %",
            data: ys,
            borderColor: "#4f7cff",
            backgroundColor: "rgba(79,124,255,0.1)",
            fill: true,
            tension: 0.2,
            pointRadius: 0,
          },
          {
            label: "Equality",
            data: xs,
            borderColor: "#454c63",
            borderDash: [4, 4],
            pointRadius: 0,
          },
          {
            label: "90% emission",
            data: markData,
            borderColor: "#f0c040",
            backgroundColor: "#f0c040",
            pointRadius: 8,
            pointHoverRadius: 10,
            showLine: false,
          },
        ],
      },
      options: {
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 100,
            title: { display: true, text: "% of cells (highest emission first)", color: "#8b91a8" },
            ticks: {
              color: "#8b91a8",
              maxRotation: 0,
              stepSize: 10,
              callback: (v) => v + "%",
            },
          },
          y: {
            title: { display: true, text: "Cumulative % of emissions", color: "#8b91a8" },
            ticks: { color: "#8b91a8" },
            min: 0,
            max: 100,
          },
        },
        plugins: {
          legend: { labels: { color: "#8b91a8" } },
          tooltip: {
            callbacks: {
              afterBody: () =>
                pct90 != null
                  ? `90% of ${pollutant} emissions from top ${pct90}% of emitting cells`
                  : "",
            },
          },
        },
      },
      plugins: pct90 != null ? [{
        id: "pct90line",
        afterDraw(chart) {
          const { ctx: c, chartArea, scales } = chart;
          const x = scales.x.getPixelForValue(pct90);
          if (x < chartArea.left || x > chartArea.right) return;
          c.save();
          c.strokeStyle = "#f0c040";
          c.lineWidth = 1.5;
          c.setLineDash([4, 3]);
          c.beginPath();
          c.moveTo(x, chartArea.top);
          c.lineTo(x, chartArea.bottom);
          c.stroke();
          c.restore();
        },
      }] : [],
    });

    const note = document.getElementById("lorenz-note");
    if (note) {
      note.textContent =
        pct90 != null
          ? `${pollutant}: 90% of emissions come from the top ${pct90}% of cells with non-zero values.`
          : `${pollutant}: not enough spatial variation to estimate a 90% point.`;
    }
  }

  function renderGini() {
    const box = document.getElementById("gini-badges");
    box.innerHTML = "";
    (data.gini || [])
      .filter((g) => g.pollutant === pollutant)
      .sort((a, b) => b.gini - a.gini)
      .forEach((g) => {
        const el = document.createElement("div");
        el.className = "gini-badge";
        el.style.borderColor = g.color;
        const loc = g.gini >= 0.7 ? "highly localized" : g.gini >= 0.4 ? "mixed" : "diffuse";
        el.innerHTML = `<strong style="color:${g.color}">${g.label}</strong> <span>${g.gini}</span> — ${loc}`;
        box.appendChild(el);
      });
    const note = document.getElementById("gini-note");
    if (note) {
      note.textContent =
        `Gini index (0 = evenly spread, 1 = concentrated) per sector for ${pollutant} area emissions.`;
    }
  }

  function renderOverview() {
    renderSummary();
    renderComposition();
    renderRadarToggles();
    renderRadar();
  }

  function renderSpatial() {
    renderHistogram();
    renderLorenz();
    renderGini();
  }

  async function open() {
    const r = await fetch(API + "/api/viz/analytics");
    const json = await r.json();
    if (json.error) throw new Error(json.error);
    data = json;
    pollutant = (data.pollutants || [])[0] || "NOx";
    const sel = document.getElementById("analytics-pollutant");
    sel.innerHTML = "";
    (data.pollutants || []).forEach((p) => {
      const o = document.createElement("option");
      o.value = p;
      o.textContent = p;
      sel.appendChild(o);
    });
    sel.value = pollutant;
    sel.onchange = () => {
      pollutant = sel.value;
      syncMapThreshold();
      renderSpatial();
    };
    destroyCharts();
    syncMapThreshold();
    renderOverview();
    renderSpatial();
  }

  return { open };
})();
