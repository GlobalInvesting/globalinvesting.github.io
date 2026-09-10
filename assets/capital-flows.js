/*
capital-flows.js  v2.0-beta — Capital Flows panel (TIC top holders + SEC
Registered Funds Flows by asset class + SEC Money Market Fund Statistics)

Reads capital-flows-data/capital_flows.json (written by
fetch_capital_flows.py). Gates each sub-panel on its own history-length
threshold — same UX pattern as the FX Fair Value "Accumulating business-day
history" progress bar — rather than showing a signal fit on too few points.

v2.0-beta: ICI removed (renderIci() deleted) — replaced with
renderRegisteredFunds(), which renders SEC's Form N-PORT-derived net-flow-
by-asset-class table using the same row layout/sticky-header/badge
conventions as renderTic(), not the older plainer table style ICI/MMF used.
See fetch_capital_flows.py v3.0's docstring for the full sourcing history.

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

  // A source that fails on a given run (network error, layout change, a
  // permanent block like ICI's — see CHANGELOG.md v8.436.0/v8.437.0, since
  // removed in v3.0) renders as `null` in the JSON. Before this helper
  // existed, a null source's early-return in renderTic()/renderRegistered
  // Funds()/renderMmf() meant renderGate() was never called at all, leaving
  // the pre-JS default HTML (the "Accumulating ... 0/12" gate div, visible
  // by default) on screen forever — indistinguishable from a source that's
  // genuinely still accumulating its first 12 points, when it's actually a
  // disclosed failure. This makes the null case an explicit third state,
  // not a silent fallthrough of the "still accumulating" one.
  function showUnavailable(prefix) {
    const gateEl = document.getElementById(`capflows-${prefix}-gate`);
    const wrap = document.getElementById(`capflows-${prefix}-wrap`);
    const unavailEl = document.getElementById(`capflows-${prefix}-unavailable`);
    if (gateEl) gateEl.style.display = "none";
    if (wrap) wrap.style.display = "none";
    if (unavailEl) unavailEl.style.display = "block";
  }

  function renderGate(prefix, gate) {
    const wrap = document.getElementById(`capflows-${prefix}-wrap`);
    const gateEl = document.getElementById(`capflows-${prefix}-gate`);
    const unavailEl = document.getElementById(`capflows-${prefix}-unavailable`);
    const progressText = document.getElementById(`capflows-${prefix}-progress-text`);
    const progressBar = document.getElementById(`capflows-${prefix}-progress-bar`);
    if (!gate) return;
    if (unavailEl) unavailEl.style.display = "none";
    const unit = "mo";
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
    if (!tic) { showUnavailable("tic"); return; }
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

  function renderRegisteredFunds(rf) {
    if (!rf) { showUnavailable("rf"); return; }
    renderGate("rf", rf.gate);
    const tbody = document.getElementById("capflows-rf-tbody");
    const asof = document.getElementById("capflows-rf-asof");
    if (!tbody) return;
    tbody.innerHTML = rf.rows
      .map((row) => {
        const momColor = row.mom_change_bn > 0 ? "var(--up)" : row.mom_change_bn < 0 ? "var(--down)" : "var(--text3)";
        return `<tr>
          <td style="padding:4px 8px 4px 16px;white-space:nowrap;">${row.category}</td>
          <td style="text-align:right;padding:4px 8px;color:${row.net_flow_bn > 0 ? "var(--up)" : row.net_flow_bn < 0 ? "var(--down)" : "var(--text3)"};">${fmtSigned(row.net_flow_bn, 1)}</td>
          <td style="text-align:right;padding:4px 8px;color:${momColor};">${fmtSigned(row.mom_change_bn, 1)}</td>
          <td style="text-align:right;padding:4px 16px 4px 8px;">${signalBadge(row.signal)}</td>
        </tr>`;
      })
      .join("");
    if (asof) asof.textContent = `As of ${rf.as_of} · SEC Registered Funds Flows (Form N-PORT, ETF + Mutual Fund combined)`;
  }

  function fmtBn(usd) {
    if (usd === null || usd === undefined) return "—";
    return (usd / 1e9).toFixed(1);
  }

  function renderMmf(mmf) {
    if (!mmf) { showUnavailable("mmf"); return; }
    renderGate("mmf", mmf.gate);
    const tbody = document.getElementById("capflows-mmf-tbody");
    const asof = document.getElementById("capflows-mmf-asof");
    if (!tbody) return;
    const rows = mmf.monthly.slice().reverse();
    tbody.innerHTML = rows
      .map((row, i) => {
        const isLatest = i === 0;
        const sig = isLatest ? mmf.signals.total : null;
        return `<tr>
          <td style="padding:4px 8px 4px 16px;white-space:nowrap;">${row.month}</td>
          <td style="text-align:right;padding:4px 8px;">${fmtBn(row.government)}</td>
          <td style="text-align:right;padding:4px 8px;">${fmtBn(row.prime)}</td>
          <td style="text-align:right;padding:4px 8px;">${fmtBn(row.tax_exempt)}</td>
          <td style="text-align:right;padding:4px 8px;">${fmtBn(row.total)}</td>
          <td style="text-align:right;padding:4px 16px 4px 8px;">${sig ? signalBadge(sig.signal) : ""}</td>
        </tr>`;
      })
      .join("");
    if (asof) asof.textContent = `As of ${mmf.as_of} · SEC Money Market Fund Statistics (Form N-MFP)`;
  }

  async function loadCapitalFlows() {
    try {
      const resp = await fetch(DATA_URL, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const doc = await resp.json();
      renderTic(doc.tic);
      renderRegisteredFunds(doc.registered_funds);
      renderMmf(doc.mmf);
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
