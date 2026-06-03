/**
 * feed.js — RSS / JSON Feed subscription popover
 * Renders a compact popover from the statusbar RSS button.
 * No external dependencies. Uses the same CSS variable palette as the terminal.
 *
 * Global: toggleRssPopover()
 */

(function () {
  'use strict';

  const FEED_XML  = 'https://globalinvesting.github.io/feed.xml';
  const FEED_JSON = 'https://globalinvesting.github.io/feed.json';

  // ── Build popover DOM once ───────────────────────────────────────────────

  function positionPopover(container) {
    const btn  = document.getElementById('rss-btn');
    const rect = btn ? btn.getBoundingClientRect() : null;
    const GAP  = 8;
    if (rect) {
      container.style.bottom = (window.innerHeight - rect.top + GAP) + 'px';
      container.style.right  = (window.innerWidth  - rect.right)     + 'px';
    } else {
      container.style.bottom = '36px';
      container.style.right  = '16px';
    }
  }

  function buildPopover(container) {
    container.style.cssText = [
      'position:fixed',
      'width:260px',
      'max-width:calc(100vw - 16px)',
      'box-sizing:border-box',
      'overflow:hidden',
      'word-wrap:break-word',
      'background:var(--head-bg,#161b22)',
      'border:1px solid var(--border,#30363d)',
      'border-radius:6px',
      'padding:12px 14px',
      'box-shadow:0 8px 24px rgba(0,0,0,.45)',
      'z-index:9999',
      'font-family:var(--font-ui,system-ui,sans-serif)',
    ].join(';');
    positionPopover(container);

    container.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
        <span style="font-size:11px;font-weight:600;letter-spacing:.06em;color:var(--blue,#4f7fff);text-transform:uppercase;">Subscribe</span>
        <button id="rss-close" aria-label="Close feed popover" style="background:none;border:none;cursor:pointer;color:var(--text-dim,#8b949e);font-size:14px;padding:0;line-height:1;">&#x2715;</button>
      </div>
      <p style="font-size:11px;color:var(--text-dim,#8b949e);margin:0 0 10px;line-height:1.5;white-space:normal;word-wrap:break-word;">
        AI market narrative &amp; signals, updated at each major session transition.
      </p>
      <div style="display:flex;flex-direction:column;gap:6px;">
        <a id="rss-link-xml"
           href="${FEED_XML}"
           target="_blank"
           rel="noopener noreferrer"
           style="display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--panel-bg,#1c2128);border:1px solid var(--border,#30363d);border-radius:4px;text-decoration:none;color:var(--text,#e6edf3);font-size:11px;font-family:var(--font-mono,monospace);transition:border-color .15s;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 11a9 9 0 0 1 9 9"/><path d="M4 4a16 16 0 0 1 16 16"/><circle cx="5" cy="19" r="1"/></svg>
          RSS 2.0 — feed.xml
        </a>
        <a id="rss-link-json"
           href="${FEED_JSON}"
           target="_blank"
           rel="noopener noreferrer"
           style="display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--panel-bg,#1c2128);border:1px solid var(--border,#30363d);border-radius:4px;text-decoration:none;color:var(--text,#e6edf3);font-size:11px;font-family:var(--font-mono,monospace);transition:border-color .15s;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
          JSON Feed 1.1 — feed.json
        </a>
      </div>
      <p style="font-size:10px;color:var(--text-dim,#8b949e);margin:10px 0 0;line-height:1.4;white-space:normal;word-wrap:break-word;">
        Paste either URL into Feedly, NetNewsWire, Reeder, or any RSS reader.
      </p>
    `;

    // Close button
    container.querySelector('#rss-close').addEventListener('click', function (e) {
      e.stopPropagation();
      closeRssPopover();
    });

    // Hover effect on links
    ['rss-link-xml', 'rss-link-json'].forEach(function (id) {
      const el = container.querySelector('#' + id);
      if (!el) return;
      el.addEventListener('mouseenter', function () {
        el.style.borderColor = 'var(--blue,#4f7fff)';
      });
      el.addEventListener('mouseleave', function () {
        el.style.borderColor = 'var(--border,#30363d)';
      });
    });
  }

  // ── Public toggle ────────────────────────────────────────────────────────

  function closeRssPopover() {
    const popover = document.getElementById('rss-popover');
    const btn     = document.getElementById('rss-btn');
    if (!popover) return;
    popover.style.display = 'none';
    if (btn) btn.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', _outsideClick);
  }

  function _outsideClick(e) {
    const anchor = document.getElementById('rss-anchor');
    if (anchor && !anchor.contains(e.target)) {
      closeRssPopover();
    }
  }

  window.toggleRssPopover = function toggleRssPopover() {
    const popover = document.getElementById('rss-popover');
    const btn     = document.getElementById('rss-btn');
    if (!popover) return;

    const isOpen = popover.style.display !== 'none';
    if (isOpen) {
      closeRssPopover();
      return;
    }

    // Build content lazily on first open
    if (!popover._built) {
      buildPopover(popover);
      popover._built = true;
    }

    // Re-position on every open so the popover tracks the button
    // correctly after any window resize since the last open.
    positionPopover(popover);
    popover.style.display = 'block';
    if (btn) btn.setAttribute('aria-expanded', 'true');

    // Close on outside click (next tick so this event doesn't immediately close it)
    setTimeout(function () {
      document.addEventListener('click', _outsideClick);
    }, 0);
  };

  // ── Escape key closes popover ────────────────────────────────────────────

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      const popover = document.getElementById('rss-popover');
      if (popover && popover.style.display !== 'none') {
        closeRssPopover();
        const btn = document.getElementById('rss-btn');
        if (btn) btn.focus();
      }
    }
  });

})();
