/* PNG export — charts via html2canvas/Chart.js, map via server-side render */
window.UrbEmExport = (function () {
  const API = "";

  async function pickPath(defaultName) {
    const r = await fetch(API + "/api/dialog/save-png", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ default_name: defaultName }),
    });
    const d = await r.json();
    if (d.cancelled || !d.path) return null;
    return d.path;
  }

  async function writePng(path, dataUrl) {
    const r = await fetch(API + "/api/export/png", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, data: dataUrl }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
  }

  async function saveDataUrl(dataUrl, defaultName) {
    const path = await pickPath(defaultName);
    if (!path) return;
    await writePng(path, dataUrl);
  }

  async function exportChart(chart, defaultName) {
    if (!chart) throw new Error("Chart not ready — wait for analytics to finish loading");
    await saveDataUrl(chart.toBase64Image("image/png", 2), defaultName);
  }

  async function exportElement(el, defaultName) {
    if (!el) return;
    if (typeof html2canvas === "undefined") {
      throw new Error("html2canvas not loaded — hard-refresh the page (Ctrl+F5)");
    }
    const canvas = await html2canvas(el, {
      backgroundColor: "#181c25",
      scale: 2,
      useCORS: true,
      allowTaint: false,
      logging: false,
      ignoreElements: (node) => node.classList && node.classList.contains("export-png-btn"),
    });
    await saveDataUrl(canvas.toDataURL("image/png"), defaultName);
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = reject;
      fr.readAsDataURL(blob);
    });
  }

  // Map export is now a clean server-side render: the backend draws the basemap,
  // emission grid and legend in Web Mercator, so the grid stays pixel-aligned
  // (no html2canvas pane baking / outline shift).
  async function exportMap(_map, params, defaultName) {
    if (!params || !params.bounds) {
      throw new Error("Map bounds unavailable — reload the results view (Ctrl+F5)");
    }
    const r = await fetch(API + "/api/viz/export-map", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bounds: params.bounds,
        pollutant: params.pollutant,
        threshold: params.threshold,
        area_sectors: params.area_sectors,
        point_sectors: params.point_sectors,
        basemap_url: params.basemap_url,
        width: params.width,
        height: params.height,
      }),
    });
    if (!r.ok) {
      let msg = `export failed (${r.status})`;
      try {
        const d = await r.json();
        if (d.error) msg = d.error;
      } catch (e) {}
      throw new Error(msg);
    }
    const blob = await r.blob();
    const dataUrl = await blobToDataUrl(blob);
    await saveDataUrl(dataUrl, defaultName);
  }

  return { exportChart, exportElement, exportMap };
})();
