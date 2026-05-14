/**
 * share.js — Narrative share ghost button
 * Triggered by the share icon that appears on #narrative hover.
 *
 * Flow:
 *   1. Reads the current narrative text + regime from the DOM.
 *   2. Builds a compact share string: regime + first sentence + URL.
 *   3. Opens native Web Share API if available (mobile); falls back to
 *      clipboard copy + 2s confirmation tooltip on desktop.
 *
 * Global: shareNarrative()
 */

(function () {
  'use strict';

  const SITE_URL = 'https://globalinvesting.github.io/';

  /**
   * Truncate text to maxLen chars, breaking at the last word boundary.
   */
  function _truncate(text, maxLen) {
    if (!text || text.length <= maxLen) return text;
    const cut = text.lastIndexOf(' ', maxLen);
    return (cut > 0 ? text.slice(0, cut) : text.slice(0, maxLen)) + '…';
  }

  /**
   * Build the share snippet (regime + first sentence, no URL appended).
   * The URL is passed separately in Web Share API calls so the OS/app handles
   * link placement — avoids double-URL on WhatsApp and similar apps that
   * automatically concatenate the `text` and `url` parameters.
   */
  function _buildShareSnippet() {
    const regimeEl = document.getElementById('narrative-regime');
    const textEl   = document.getElementById('narrative-text');

    const regime  = (regimeEl ? regimeEl.textContent.trim() : '').replace(/^__STALE__/, '');
    const narr    = textEl ? textEl.textContent.trim() : '';

    if (!narr || narr === 'Loading market narrative…') return null;

    // First sentence (up to first period + space, otherwise first 200 chars)
    const dotIdx = narr.search(/\.\s/);
    const snippet = dotIdx > 0 && dotIdx < 220
      ? narr.slice(0, dotIdx + 1)
      : _truncate(narr, 200);

    const regimePart = regime && regime !== '—' ? `[${regime}] ` : '';
    return `${regimePart}${snippet}`;
  }

  /**
   * Build the full share string for clipboard/legacy copy.
   * Appends the URL so plain-text recipients get the link.
   */
  function _buildShareText() {
    const snippet = _buildShareSnippet();
    return snippet ? `${snippet}\n\n${SITE_URL}` : null;
  }

  /**
   * Show a 2-second "Copied" confirmation on the button.
   */
  function _showCopied(btn) {
    const original = btn.innerHTML;
    btn.classList.add('copied');
    btn.setAttribute('aria-label', 'Copied!');
    btn.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>';
    setTimeout(function () {
      btn.classList.remove('copied');
      btn.setAttribute('aria-label', 'Share market narrative');
      btn.innerHTML = original;
    }, 2000);
  }

  /**
   * Main share handler — attached to window for inline onclick.
   */
  window.shareNarrative = function shareNarrative() {
    const text = _buildShareText();
    if (!text) return;

    const btn = document.getElementById('narr-share-btn');

    // Web Share API — available on mobile browsers and some desktop Chromium.
    // Pass the narrative snippet as `text` and the site URL as `url` separately.
    // Do NOT include SITE_URL inside `text` — the OS concatenates text + url
    // automatically (e.g. WhatsApp, iMessage), so embedding it in text would
    // produce a duplicate link. Email clients display both fields correctly this way.
    if (navigator.share) {
      var snippet = _buildShareSnippet();
      if (!snippet) return;
      navigator.share({
        title: 'Global Investing FX — Market Narrative',
        text:  snippet,
        url:   SITE_URL,
      }).catch(function () {
        // User cancelled — no action needed
      });
      return;
    }

    // Clipboard fallback
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        if (btn) _showCopied(btn);
      }).catch(function () {
        _legacyCopy(text, btn);
      });
    } else {
      _legacyCopy(text, btn);
    }
  };

  /**
   * Legacy execCommand copy for older browsers.
   */
  function _legacyCopy(text, btn) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;left:-9999px;top:-9999px;opacity:0;';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      document.execCommand('copy');
      if (btn) _showCopied(btn);
    } catch (_) {
      // Silent fail — clipboard unavailable
    } finally {
      document.body.removeChild(ta);
    }
  }

})();
