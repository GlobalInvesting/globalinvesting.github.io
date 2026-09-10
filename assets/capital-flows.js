/*
capital-flows.js  v2.1-beta — Capital Flows panel (TIC top holders +
combined SEC Registered Funds Flows + SEC Money Market Fund Statistics)

Reads capital-flows-data/capital_flows.json (written by
fetch_capital_flows.py). Gates each sub-panel on its own history-length
threshold — same UX pattern as the FX Fair Value "Accumulating business-day
history" progress bar — rather than showing a signal fit on too few points.

v2.1-beta: renderRegisteredFunds() + renderMmf() merged into one
renderFundFlows(), writing registered_funds.rows and mmf.rows (mmf.rows
added in fetch_capital_flows.py v3.1) into a SINGLE table, one row per
asset class/category, matching renderTic()'s row layout throughout. A
muted section-header row is injected between the two groups because the
two sources have genuinely different publication vintages (Registered
Funds ≈ real-time for the prior month; MMF runs ~2mo behind) — combining
the tables' LAYOUT is a real fix (both are "one row per category" data,
they just used to render differently), but combining them into one
undifferentiated series would hide that vintage difference, so each
section still discloses its own "as of" date. Each group also gates
independently: if one source's gate isn't ready, that group's row is a
single "Accumulating history — n/12mo" line instead of hiding the whole
table (see showUnavailable()'s doc for why a null/not-ready source must
never look identical to "still loading").

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

  // Shared row renderer for both Registered Funds and MMF sections — both
  // now emit the identical {category, net_flow_bn, mom_change_bn, z,
  // signal} row shape (see fetch_capital_flows.py v3.1), so one renderer
  // covers both instead of duplicating the markup per source.
  function fundFlowRows(rows) {
    return rows
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
  }

  // Muted divider row between the Registered Funds and MMF groups within
  // the one combined table — this is where each source's own "as of"
  // vintage is disclosed, since the two reports are genuinely not on the
  // same publication schedule (see capital-flows.js's top-of-file note).
  function fundFlowSectionRow(label) {
    return `<tr><td colspan="4" style="padding:10px 16px 4px;font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.03em;">${label}</td></tr>`;
  }

  // In-table equivalent of renderGate()'s progress bar, used when a group
  // (Registered Funds or MMF) isn't past its own history gate yet — a
  // single spanning row instead of hiding the whole combined table, since
  // the other group may already be ready.
  function fundFlowGateRow(gate) {
    const have = gate ? gate.have : 0;
    const need = gate ? gate.need : 12;
    const pct = Math.min(100, (have / need) * 100);
    return `<tr><td colspan="4" style="padding:6px 16px 12px;font-size:11px;color:var(--text2);font-family:var(--font-ui);">
      Accumulating monthly history — ${have}/${need}mo
      <div style="height:4px;background:var(--bg2);border-radius:2px;margin-top:6px;overflow:hidden;">
        <div style="height:100%;width:${pct}%;background:var(--accent);border-radius:2px;"></div>
      </div>
    </td></tr>`;
  }

  function fundFlowUnavailableRow() {
    return `<tr><td colspan="4" style="padding:6px 16px 12px;font-size:11px;color:var(--text3);font-family:var(--font-ui);">Currently unavailable.</td></tr>`;
  }

  // Combined Registered Funds + MMF table — see the v2.1-beta note at the
  // top of this file for why these two are one table now (same row shape)
  // but still two clearly-labeled sections (different vintages).
  function renderFundFlows(rf, mmf) {
    const tbody = document.getElementById("capflows-flows-tbody");
    const wrap = document.getElementById("capflows-flows-wrap");
    const unavailEl = document.getElementById("capflows-flows-unavailable");
    const asof = document.getElementById("capflows-flows-asof");
    if (!tbody) return;

    if (!rf && !mmf) {
      if (wrap) wrap.style.display = "none";
      if (unavailEl) unavailEl.style.display = "block";
      return;
    }
    if (unavailEl) unavailEl.style.display = "none";
    if (wrap) wrap.style.display = "block";

    const asofParts = [];
    let html = "";

    if (rf) {
      html += fundFlowSectionRow(`Registered fund net flows by asset class · SEC (Form N-PORT) · as of ${rf.as_of}`);
      html += rf.gate && rf.gate.ready ? fundFlowRows(rf.rows) : fundFlowGateRow(rf.gate);
      asofParts.push(`Registered Funds as of ${rf.as_of}`);
    } else {
      html += fundFlowSectionRow("Registered fund net flows by asset class · SEC (Form N-PORT)");
      html += fundFlowUnavailableRow();
    }

    html += fundFlowSectionRow(
      mmf
        ? `Money market fund flows · SEC (Form N-MFP) · as of ${mmf.as_of} · ~2mo lag`
        : "Money market fund flows · SEC (Form N-MFP)"
    );
    if (mmf) {
      html += mmf.gate && mmf.gate.ready ? fundFlowRows(mmf.rows) : fundFlowGateRow(mmf.gate);
      asofParts.push(`MMF as of ${mmf.as_of}`);
    } else {
      html += fundFlowUnavailableRow();
    }

    tbody.innerHTML = html;
    if (asof) asof.textContent = asofParts.join(" · ");
  }

  async function loadCapitalFlows() {
    try {
      const resp = await fetch(DATA_URL, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const doc = await resp.json();
      renderTic(doc.tic);
      renderFundFlows(doc.registered_funds, doc.mmf);
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
