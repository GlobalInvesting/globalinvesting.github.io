(function () {
  'use strict';

  var overlay, input, resultsEl, lastFocused, activeIndex;
  activeIndex = -1;

  function pairIds() {
    return (Array.isArray(window.PAIRS) ? window.PAIRS : []).map(function (p) { return p.id; });
  }

  function pairLabel(id) {
    return id.slice(0, 3).toUpperCase() + '/' + id.slice(3).toUpperCase();
  }

  function normalize(s) {
    return (s || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  }

  function buildSectionIndex() {
    var out = [];
    document.querySelectorAll('.top-nav a[data-target]').forEach(function (a) {
      out.push({ type: 'section', target: a.getAttribute('data-target'), label: (a.textContent || '').trim() });
    });
    return out;
  }

  function matches(query) {
    var q = normalize(query);
    var out = [];
    if (!q) return out;
    pairIds().forEach(function (id) {
      if (normalize(id).indexOf(q) !== -1) {
        out.push({ type: 'pair', id: id, label: pairLabel(id) });
      }
    });
    buildSectionIndex().forEach(function (s) {
      if (normalize(s.label).indexOf(q) !== -1 || normalize(s.target).indexOf(q) !== -1) {
        out.push(s);
      }
    });
    return out.slice(0, 8);
  }

  function goToSection(target) {
    var link = document.querySelector('.top-nav a[data-target="' + target + '"]');
    if (link) {
      link.click();
      return;
    }
    var el = document.getElementById(target);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function closeOtherOpenDetails(exceptRow) {
    var openMajor = document.querySelector('#fx-pairs-tbody tr.pd-selected');
    if (openMajor && openMajor !== exceptRow && typeof window.toggleInlineDetail === 'function') {
      window.toggleInlineDetail(openMajor);
    }
    var openCross = document.querySelector('#sidebar .sb-row.sb-selected');
    if (openCross && openCross !== exceptRow && typeof window.toggleSidebarDetail === 'function') {
      window.toggleSidebarDetail(openCross);
    }
  }

  function goToPair(id) {
    var sym = 'FX_IDC:' + id.toUpperCase();
    var row = document.querySelector('#fx-pairs-tbody tr[data-sym="' + sym + '"]') ||
      document.querySelector('#sidebar .sb-row[data-sym="' + sym + '"]');
    if (!row) return;
    closeOtherOpenDetails(row);
    row.click();
    row.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderResults(query) {
    var items = matches(query);
    activeIndex = items.length ? 0 : -1;
    if (!items.length) {
      resultsEl.innerHTML = query
        ? '<div class="cmdbar-empty">No result</div>'
        : '<div class="cmdbar-hint">Type a symbol (e.g. EURUSD) or a section name.</div>';
      resultsEl._items = [];
      return;
    }
    resultsEl.innerHTML = items.map(function (it, i) {
      var sub = it.type === 'pair' ? 'FX Pair' : 'Section';
      return '<button type="button" class="cmdbar-item' + (i === 0 ? ' active' : '') + '" data-idx="' + i +
        '" role="option" aria-selected="' + (i === 0) + '">' +
        '<span class="cmdbar-item-label">' + it.label + '</span>' +
        '<span class="cmdbar-item-sub">' + sub + '</span></button>';
    }).join('');
    resultsEl._items = items;
  }

  function selectIndex(i) {
    var items = resultsEl._items || [];
    var it = items[i];
    if (!it) return;
    closeBar();
    if (it.type === 'pair') goToPair(it.id);
    else goToSection(it.target);
  }

  function moveActive(delta) {
    var buttons = resultsEl.querySelectorAll('.cmdbar-item');
    if (!buttons.length) return;
    activeIndex = (activeIndex + delta + buttons.length) % buttons.length;
    buttons.forEach(function (b, i) {
      var on = i === activeIndex;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    buttons[activeIndex].scrollIntoView({ block: 'nearest' });
  }

  function openBar() {
    lastFocused = document.activeElement;
    overlay.hidden = false;
    document.body.style.overflow = 'hidden';
    input.value = '';
    renderResults('');
    input.focus();
  }

  function closeBar() {
    overlay.hidden = true;
    document.body.style.overflow = '';
    if (lastFocused && typeof lastFocused.focus === 'function') lastFocused.focus();
  }

  function buildOverlay() {
    overlay = document.createElement('div');
    overlay.id = 'cmdbar-overlay';
    overlay.hidden = true;
    overlay.innerHTML =
      '<div id="cmdbar-panel" role="dialog" aria-modal="true" aria-label="Command bar">' +
      '<div id="cmdbar-inputwrap">' +
      '<span id="cmdbar-caret" aria-hidden="true">&gt;</span>' +
      '<input id="cmdbar-input" type="text" autocomplete="off" spellcheck="false" ' +
      'placeholder="Type a symbol or a section name" aria-label="Command input" role="combobox" ' +
      'aria-expanded="true" aria-controls="cmdbar-results">' +
      '<span id="cmdbar-esc">Esc</span>' +
      '</div>' +
      '<div id="cmdbar-results" role="listbox" aria-label="Results"></div>' +
      '</div>';
    document.body.appendChild(overlay);
    input = overlay.querySelector('#cmdbar-input');
    resultsEl = overlay.querySelector('#cmdbar-results');

    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) closeBar();
    });
    resultsEl.addEventListener('click', function (e) {
      var btn = e.target.closest('.cmdbar-item');
      if (btn) selectIndex(parseInt(btn.getAttribute('data-idx'), 10));
    });
    input.addEventListener('input', function () {
      renderResults(input.value);
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeBar();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        moveActive(1);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        moveActive(-1);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        selectIndex(activeIndex);
      }
    });
  }

  function buildTrigger() {
    var right = document.querySelector('.topbar-right');
    if (!right) return;
    var btn = document.createElement('button');
    btn.id = 'cmdbar-trigger';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Open command bar');
    btn.title = 'Command bar (Ctrl+K)';
    btn.textContent = '\u2318K';
    btn.addEventListener('click', openBar);
    right.insertBefore(btn, right.firstChild);
  }

  document.addEventListener('keydown', function (e) {
    var k = e.key ? e.key.toLowerCase() : '';
    if ((e.ctrlKey || e.metaKey) && k === 'k') {
      e.preventDefault();
      if (overlay && !overlay.hidden) closeBar();
      else openBar();
    }
  });

  document.addEventListener('DOMContentLoaded', function () {
    buildOverlay();
    buildTrigger();
  });
})();
