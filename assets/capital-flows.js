/*
capital-flows.js  v2.2 — Capital Flows panel (TIC top holders + combined
SEC Registered Funds Flows + SEC Money Market Fund Statistics)

Reads capital-flows-data/capital_flows.json (written by
fetch_capital_flows.py). Gates each sub-panel on its own history-length
threshold — same UX pattern as the FX Fair Value "Accumulating business-day
history" progress bar — rather than showing a signal fit on too few points.

Production panel: all three gates (TIC, RF, MMF) clear with real
accumulated data (13/12, 84/12, 117/12 months respectively). Lives in
index.html between #section-econmap and #section-fair-value; this file is
a single shared module referenced by one production entry point, not
duplicated staging markup two pages could independently drift out of sync
with. No dashboard.js/dashboard.css changes were needed —
loadCapitalFlows() self-registers on DOMContentLoaded and every CSS
variable/class the panel uses (--text2/--text3/--accent/--font-ui/
--font-mono/--bg2/--border, .panel-head/.panel-title/.panel-sub) already
exists in the production stylesheet, since the whole panel was styled to
match sibling production panels' conventions from the start.

v2.2: fund-flows table flattened to a single plain list, matching how
ICI's own weekly "Combined Estimated Long-Term Flows" release (the
industry reference this project targets) lists categories — Domestic
Equity, World Equity, Hybrid, Taxable Bond, Municipal Bond, Money Market —
as one undifferentiated row list, with Money Market as one line among the
others rather than its own sub-table. The muted section-header rows are
removed; Registered Funds' rows (registered_funds.rows) render first, then
a single "Money Market" row (mmf.rows' own "total" category, relabeled) is
appended last — a fixed position rather than interleaved, since it already
aggregates three sub-categories (Government/Prime/Tax-exempt) into one
figure and reads better as a summary line than mixed in with individual
asset classes. The Government/Prime/Tax-exempt breakdown itself is dropped
from the visible table (drill-down detail, not a top-level "flows by
category" row) but stays in the underlying JSON (mmf.rows, unchanged) for
a future per-category view.
Vintage disclosure — the actual reason section headers existed in the
first place (Registered Funds ≈ real-time for the prior month; MMF runs
~2mo behind, so the two groups sit on different "as of" dates) — is kept,
just moved to the existing single-line footer beneath the table
("Registered Funds as of ... · MMF as of ..."), which already disclosed
both dates before this change; nothing about that honesty guarantee is
lost by removing the in-table headers. A source-specific gate/unavailable
row (fundFlowGateRow()/fundFlowUnavailableRow()) still names which source
it refers to, since there's no longer a header above it to say so.

v2.0: ICI removed (renderIci() deleted) — replaced with
renderRegisteredFunds(), which renders SEC's Form N-PORT-derived net-flow-
by-asset-class table using the same row layout/sticky-header/badge
conventions as renderTic(), not the older plainer table style ICI/MMF used.
See fetch_capital_flows.py v3.0's docstring for the full sourcing history.
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

  // In-table equivalent of renderGate()'s progress bar, used when a group
  // (Registered Funds or Money Market) isn't past its own history gate
  // yet — a single spanning row instead of hiding the whole flat list,
  // since the other group may already be ready. Names the source
  // explicitly since v2.2-beta removed the section header that used to
  // say so.
  function fundFlowGateRow(gate, label) {
    const have = gate ? gate.have : 0;
    const need = gate ? gate.need : 12;
    const pct = Math.min(100, (have / need) * 100);
    return `<tr><td colspan="4" style="padding:6px 16px 12px;font-size:11px;color:var(--text2);font-family:var(--font-ui);">
      Accumulating ${label} history — ${have}/${need}mo
      <div style="height:4px;background:var(--bg2);border-radius:2px;margin-top:6px;overflow:hidden;">
        <div style="height:100%;width:${pct}%;background:var(--accent);border-radius:2px;"></div>
      </div>
    </td></tr>`;
  }

  function fundFlowUnavailableRow(label) {
    return `<tr><td colspan="4" style="padding:6px 16px 12px;font-size:11px;color:var(--text3);font-family:var(--font-ui);">${label} currently unavailable.</td></tr>`;
  }

  // Flat fund-flows list — Registered Funds' own rows first, then a single
  // "Money Market" summary row appended last (mmf.rows' "total" category,
  // relabeled), matching how ICI's own weekly release lists categories:
  // one plain list, no sub-table. See the v2.2-beta note at the top of
  // this file for why Money Market is a fixed trailing row rather than
  // interleaved, and why the Government/Prime/Tax-exempt breakdown isn't
  // shown here. Each source's own "as of" vintage is disclosed in the
  // footer line below the table (asofParts), not in the row list itself.
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
      html += rf.gate && rf.gate.ready ? fundFlowRows(rf.rows) : fundFlowGateRow(rf.gate, "registered fund flows");
      asofParts.push(`Registered Funds as of ${rf.as_of}`);
    } else {
      html += fundFlowUnavailableRow("Registered fund flows");
    }

    if (mmf) {
      if (mmf.gate && mmf.gate.ready) {
        const total = mmf.rows.find((row) => row.category === "Money Market — Total");
        html += total ? fundFlowRows([{ ...total, category: "Money Market" }]) : fundFlowUnavailableRow("Money market flows");
      } else {
        html += fundFlowGateRow(mmf.gate, "money market");
      }
      asofParts.push(`MMF as of ${mmf.as_of}`);
    } else {
      html += fundFlowUnavailableRow("Money market flows");
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
