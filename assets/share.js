
(function () {
  'use strict';

  const SITE_URL = 'https://globalinvesting.github.io/';

  function _truncate(text, maxLen) {
    if (!text || text.length <= maxLen) return text;
    const cut = text.lastIndexOf(' ', maxLen);
    return (cut > 0 ? text.slice(0, cut) : text.slice(0, maxLen)) + '…';
  }

  function _buildShareSnippet() {
    const regimeEl = document.getElementById('narrative-regime');
    const textEl   = document.getElementById('narrative-text');

    const regime  = (regimeEl ? regimeEl.textContent.trim() : '').replace(/^__STALE__/, '');
    const narr    = textEl ? textEl.textContent.trim() : '';

    if (!narr || narr === 'Loading market narrative…') return null;

    const dotIdx = narr.search(/\.\s/);
    const snippet = dotIdx > 0 && dotIdx < 220
      ? narr.slice(0, dotIdx + 1)
      : _truncate(narr, 200);

    const regimePart = regime && regime !== '—' ? `[${regime}] ` : '';
    return `${regimePart}${snippet}`;
  }

  function _buildShareText() {
    const snippet = _buildShareSnippet();
    return snippet ? `${snippet}\n\n${SITE_URL}` : null;
  }

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

  window.shareNarrative = function shareNarrative() {
    const text = _buildShareText();
    if (!text) return;

    const btn = document.getElementById('narr-share-btn');

    if (navigator.share) {
      var snippet = _buildShareSnippet();
      if (!snippet) return;
      navigator.share({
        title: 'Global Investing FX — Market Narrative',
        text:  snippet,
        url:   SITE_URL,
      }).catch(function () {
      });
      return;
    }

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
    } finally {
      document.body.removeChild(ta);
    }
  }

})();
