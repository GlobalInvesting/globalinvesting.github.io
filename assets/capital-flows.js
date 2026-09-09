/*
capital-flows.js  v1.0-beta — Capital Flows panel (TIC top holders + ICI
weekly fund flows)

Reads capital-flows-data/capital_flows.json (written by
fetch_capital_flows.py). Gates each sub-panel on its own history-length
threshold — same UX pattern as the FX Fair Value "Accumulating business-day
history" progress bar — rather than showing a signal fit on too few points.

Beta-stage note: this file is wired into index-beta.html only. Once the
panel is confirmed visually and the data pipeline has run long enough to
clear both gates with real data, this section (HTML + this script + the
JSON path) gets promoted into index.html / dashboard.css per the usual
review flow — see CHANGELOG.md.
*/

(function () {
  const DATA_URL = "capital-flows-data/capital_flows.json";

  function signalColor(signal) {
    if (signal === "Accumulating") return "var(--up)";
    if (signal === "Reducing") return "var(--down)";
    return "var(--text3)";
  }

  function signalBadge(signal) {
    if (!signal) return '<span style="color:var(--text3);">—</span>';
    const color = signalColor(signal);
    return `<span style="display:inline-block;padding:2px 7px;border-radius:3px;background:${color}22;color:${color};font-size:10px;">${signal}</span>`;
  }

  function fmtSigned(v, decimals) {
    if (v === null || v === undefined) return "—";
    const s = v.toFixed(decimals);
    return v > 0 ? "+" + s : s;
  }

  function renderGate(prefix, gate) {
    const wrap = document.getElementById(`capflows-${prefix}-wrap`);
    const gateEl = document.getElementById(`capflows-${prefix}-gate`);
    const progressText = document.getElementById(`capflows-${prefix}-progress-text`);
    const progressBar = document.getElementById(`capflows-${prefix}-progress-bar`);
    if (!gate) return;
    const unit = prefix === "tic" ? "mo" : "wk";
    if (progressText) progressText.textContent = `${gate.have}/${gate.need}${unit}`;
    if (progressBar) progressBar.style.width = `${Math.min(100, (gate.have / gate.need) * 100)}%`;
    if (gate.ready) {
      if (gateEl) gateEl.style.display = "none";
      if (wrap) wrap.style.display = "block";
    } else {
      if (gateEl) gateEl.style.display = "block";
      if (wrap) wrap.style.display = "none";
    }
  }

  function renderTic(tic) {
    if (!tic) return;
    renderGate("tic", tic.gate);
    const tbody = document.getElementById("capflows-tic-tbody");
    const asof = document.getElementById("capflows-tic-asof");
    if (!tbody) return;
    tbody.innerHTML = tic.top10
      .map((row) => {
        const flag = row.iso2
          ? `<span class="fi fi-${row.iso2}" style="margin-right:5px;border-radius:2px;"></span>`
          : "";
        return `<tr>
          <td style="padding:4px 8px 4px 16px;white-space:nowrap;">${flag}${row.country}</td>
          <td style="text-align:right;padding:4px 8px;">${row.holdings_bn.toFixed(1)}</td>
          <td style="text-align:right;padding:4px 8px;color:${row.mom_change_bn > 0 ? "var(--up)" : row.mom_change_bn < 0 ? "var(--down)" : "var(--text3)"};">${fmtSigned(row.mom_change_bn, 1)}</td>
          <td style="text-align:right;padding:4px 16px 4px 8px;">${signalBadge(row.signal)}</td>
        </tr>`;
      })
      .join("");
    if (asof) asof.textContent = `As of ${tic.as_of} · US Treasury TIC`;
  }

  function renderIci(ici) {
    if (!ici) return;
    renderGate("ici", ici.gate);
    const tbody = document.getElementById("capflows-ici-tbody");
    const asof = document.getElementById("capflows-ici-asof");
    if (!tbody) return;
    const rows = ici.weekly.slice().reverse();
    tbody.innerHTML = rows
      .map((row, i) => {
        const isLatest = i === 0;
        const sig = isLatest ? ici.latest_signal : null;
        return `<tr>
          <td style="padding:4px 8px 4px 16px;white-space:nowrap;">${row.week_ending}</td>
          <td style="text-align:right;padding:4px 8px;">${row.equity !== null ? (row.equity / 1000).toFixed(2) : "—"}</td>
          <td style="text-align:right;padding:4px 8px;">${row.bond !== null ? (row.bond / 1000).toFixed(2) : "—"}</td>
          <td style="text-align:right;padding:4px 8px;">${row.total !== null ? (row.total / 1000).toFixed(2) : "—"}</td>
          <td style="text-align:right;padding:4px 16px 4px 8px;">${sig ? signalBadge(sig.signal) : ""}</td>
        </tr>`;
      })
      .join("");
    if (asof) asof.textContent = `Week ending ${ici.as_of} · ICI, excludes money market funds`;
  }

  async function loadCapitalFlows() {
    try {
      const resp = await fetch(DATA_URL, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const doc = await resp.json();
      renderTic(doc.tic);
      renderIci(doc.ici);
    } catch (err) {
      const ticBody = document.getElementById("capflows-tic-tbody");
      if (ticBody) {
        ticBody.innerHTML = `<tr><td colspan="4" style="padding:14px 16px;color:var(--text3);">Capital flows data unavailable.</td></tr>`;
      }
      console.error("[capital-flows] load failed:", err);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadCapitalFlows);
  } else {
    loadCapitalFlows();
  }
})();
