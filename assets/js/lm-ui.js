/* ==========================================================================
   Language Modelling — site chrome (vanilla JS, no dependencies)
   theme · navigator + read state · token-sequence masthead · KV-strip
   progress + token meter · margin index · plates/figures numbering ·
   code copy · command palette. Everything degrades to plain HTML.
   ========================================================================== */
(function () {
  'use strict';
  var LM = window.LM || { base: '', path: '/', n: '', home: false };
  var doc = document.documentElement;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var store = {
    get: function (k, d) { try { var v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set: function (k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} }
  };
  var norm = function (p) { return (p || '/').replace(/\.html$/, '').replace(/\/$/, '') || '/'; };
  var here = norm(LM.path);

  // ------------------------------------------------------------ theme
  var themeBtn = $('#themeToggle');
  if (themeBtn) themeBtn.addEventListener('click', function () {
    var t = doc.getAttribute('data-theme') === 'night' ? 'paper' : 'night';
    doc.setAttribute('data-theme', t);
    try { localStorage.setItem('lm-theme', t); } catch (e) {}
    document.dispatchEvent(new CustomEvent('lm:theme', { detail: t }));
  });

  // -------------------------------------------------------- navigator
  var navBtn = $('#navToggle'), stream = $('#stream'), scrim = $('#scrim');
  var mobile = function () { return window.innerWidth <= 1100; };
  if (!mobile() && store.get('lm-stream-closed', false)) document.body.classList.add('stream-closed');
  function setOpen(open) {
    if (mobile()) {
      document.body.classList.toggle('stream-open', open);
      if (scrim) scrim.hidden = !open;
    } else {
      document.body.classList.toggle('stream-closed', !open);
      store.set('lm-stream-closed', !open);
    }
    if (navBtn) navBtn.setAttribute('aria-expanded', String(open));
  }
  if (navBtn) navBtn.addEventListener('click', function () {
    var open = mobile() ? !document.body.classList.contains('stream-open') : document.body.classList.contains('stream-closed');
    setOpen(open);
  });
  if (scrim) scrim.addEventListener('click', function () { setOpen(false); });
  var cur = $('.sl-item.is-here');
  if (cur && stream) { var list = $('#streamList'); if (list) list.scrollTop = Math.max(0, cur.offsetTop - list.clientHeight / 2); }

  var filter = $('#navFilter');
  if (filter) filter.addEventListener('input', function () {
    var q = filter.value.trim().toLowerCase(), part = null, partHit = false;
    $$('#streamList > li').forEach(function (li) {
      if (li.classList.contains('sl-part')) {
        if (part) part.classList.toggle('filtered-out', q !== '' && !partHit);
        part = li; partHit = false; return;
      }
      var hit = q === '' || li.textContent.toLowerCase().indexOf(q) !== -1;
      li.classList.toggle('filtered-out', !hit);
      if (hit) partHit = true;
    });
    if (part) part.classList.toggle('filtered-out', q !== '' && !partHit);
  });
  if (filter) filter.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { var a = $('#streamList .sl-item:not(.filtered-out) a'); if (a) location.href = a.href; }
    if (e.key === 'Escape') { filter.value = ''; filter.dispatchEvent(new Event('input')); filter.blur(); }
  });

  // read state: chapters scrolled past 70% are marked as read
  var read = store.get('lm-read', {});
  function paintRead() {
    $$('[data-path]').forEach(function (a) {
      var p = norm(a.getAttribute('data-path'));
      var li = a.closest('.sl-item');
      if (read[p]) { (li || a).classList.add('is-read'); }
    });
  }
  paintRead();

  // ------------------------------------------------ token-sequence title
  // A decorative segmentation of the page title into subword-like pieces,
  // labelled with their sequence positions (t = 0, 1, 2, …).
  function pieces(text) {
    var out = [];
    (text.match(/\s?[^\s]+/g) || []).forEach(function (w) {
      var lead = w[0] === ' ' ? ' ' : '', core = lead ? w.slice(1) : w;
      var m = core.match(/^([\W_]*)(.*?)([\W_]*)$/);
      if (m[1]) { out.push(lead + m[1]); lead = ''; }
      var body = m[2];
      while (body.length > 7) {                 // split long words at a vowel-consonant boundary
        var cut = 4;
        for (var i = 3; i < Math.min(7, body.length - 2); i++) {
          if (/[aeiouy]/i.test(body[i]) && !/[aeiouy]/i.test(body[i + 1])) { cut = i + 2; break; }
        }
        out.push(lead + body.slice(0, cut)); lead = ''; body = body.slice(cut);
      }
      if (body) out.push(lead + body);
      if (m[3]) out.push(m[3]);
    });
    return out;
  }
  var seq = $('#mastSeq');
  if (seq) {
    var title = LM.home ? 'The cat sat on the' : (LM.title || document.title.split(' — ')[0]);
    pieces(title).slice(0, 14).forEach(function (p, i) {
      var s = document.createElement('span');
      s.className = 'tok'; s.style.animationDelay = (i * 45) + 'ms';
      s.textContent = p.replace(/^ /, '·');
      var sub = document.createElement('sub'); sub.textContent = 't' + i; s.appendChild(sub);
      seq.appendChild(s);
    });
  }

  // --------------------------------------------------- content polish
  var content = $('#content');
  if (content) {
    // breadcrumb paragraphs ("← Back to Table of Contents") become quiet links
    $$('p', content).forEach(function (p) {
      var a = p.querySelector('a');
      if (a && p.children.length <= 3 && /^←\s*Back to Table of Contents/i.test(p.textContent.trim())) p.classList.add('backlink');
    });
    // h1 "Chapter 4 — Title" → eyebrow + title
    var h1 = $('h1', content);
    if (h1) {
      var mm = h1.textContent.match(/^((?:Chapter|Appendix)\s+[\w]+)\s+[—–-]\s+(.*)$/);
      if (mm) { h1.innerHTML = '<span class="h1-ch">' + mm[1] + '</span>' + mm[2].replace(/</g, '&lt;'); }
    }
    // the dek
    $$('blockquote', content).forEach(function (q) {
      var s = q.querySelector('strong');
      if (s && /one-sentence version/i.test(s.textContent)) q.classList.add('dek');
    });
    // tables
    $$('table', content).forEach(function (t) {
      if (t.parentNode.classList && t.parentNode.classList.contains('table-wrap')) return;
      var w = document.createElement('div'); w.className = 'table-wrap';
      var h = t.previousElementSibling;
      while (h && !/^H[23]$/.test(h.tagName)) h = h.previousElementSibling;
      if (h && /^References/i.test(h.textContent)) w.classList.add('refs');
      t.parentNode.insertBefore(w, t); w.appendChild(t);
    });
    // key takeaways
    $$('h2', content).forEach(function (h) {
      if (/^Key Takeaways/i.test(h.textContent)) {
        var n = h.nextElementSibling; if (n && n.tagName === 'UL') n.classList.add('takeaways');
      }
    });
    // plate and figure numbers
    var ch = (LM.n && LM.n !== '∅' && LM.n !== '⚗') ? LM.n.replace(/^0/, '') : '';
    $$('.diagram-title', content).forEach(function (t, i) {
      var s = document.createElement('span'); s.className = 'plate-no';
      s.textContent = 'Plate ' + (ch ? ch + '.' : '') + (i + 1);
      t.insertBefore(s, t.firstChild);
    });
    $$('.img-caption figcaption, figure figcaption', content).forEach(function (c, i) {
      var s = document.createElement('span'); s.className = 'fig-no';
      s.textContent = 'Fig. ' + (ch ? ch + '.' : '') + (i + 1);
      c.insertBefore(s, c.firstChild);
    });
    // heading anchors
    $$('h2[id], h3[id]', content).forEach(function (h) {
      var a = document.createElement('a'); a.className = 'anchor'; a.href = '#' + h.id; a.textContent = '#';
      a.setAttribute('aria-label', 'Link to this section'); h.appendChild(a);
    });
    // code blocks: language label + copy
    $$('div.highlighter-rouge', content).forEach(function (b) {
      var m = (b.className.match(/language-([\w+-]+)/) || [])[1] || 'text';
      var names = { python: 'Python', bash: 'Shell', sh: 'Shell', c: 'CUDA C', cpp: 'C++', json: 'JSON', text: 'Output', plaintext: 'Text', yaml: 'YAML', javascript: 'JavaScript', js: 'JavaScript' };
      b.setAttribute('data-lang', names[m] || m);
      var pre = b.querySelector('pre'); if (!pre) return;
      var btn = document.createElement('button'); btn.type = 'button'; btn.className = 'copy-btn'; btn.textContent = 'copy';
      btn.addEventListener('click', function () {
        var txt = pre.innerText.replace(/\n$/, '');
        var ok = function () { btn.textContent = 'copied'; btn.classList.add('ok'); setTimeout(function () { btn.textContent = 'copy'; btn.classList.remove('ok'); }, 1400); };
        if (navigator.clipboard) navigator.clipboard.writeText(txt).then(ok, function () {}); else ok();
      });
      b.appendChild(btn);
    });
  }

  // ------------------------------------- KV strip, token meter, index
  var h2s = content ? $$('h2', content) : [];
  var strip = $('#kvStrip'), tokRead = $('#tokRead');
  var totalWords = content ? (content.innerText.match(/\S+/g) || []).length : 0;
  var cells = [];
  if (strip && h2s.length && !LM.home) {
    var bounds = h2s.map(function (h) { return h; });
    bounds.forEach(function (h, i) {
      var c = document.createElement('div'); c.className = 'kv-cell';
      c.title = '§' + (i + 1) + ' ' + h.textContent.replace(/#$/, '').trim();
      c.appendChild(document.createElement('i'));
      c.addEventListener('click', function () { h.scrollIntoView({ behavior: 'smooth' }); });
      strip.appendChild(c); cells.push(c);
    });
    // widths proportional to section length
    var tops = h2s.map(function (h) { return h.offsetTop; });
    var endY = content.offsetTop + content.offsetHeight;
    tops.forEach(function (t, i) { cells[i].style.flex = Math.max(1, (i + 1 < tops.length ? tops[i + 1] : endY) - t) + ' 1 0'; });
  }
  var mi = $('#marginIndex'), miList = $('#miList'), miLinks = [];
  if (mi && miList && content && !LM.home) {
    var secs = $$('h2[id], h3[id]', content);
    if (secs.length > 2) {
      var k = 0;
      secs.forEach(function (h) {
        var li = document.createElement('li'), a = document.createElement('a');
        a.href = '#' + h.id; a.className = h.tagName.toLowerCase(); a.dataset.id = h.id;
        var label = h.textContent.replace(/#$/, '').trim();
        if (h.tagName === 'H2') { k++; a.innerHTML = '<span class="mi-n">§' + k + '</span><span></span>'; a.lastChild.textContent = label; }
        else { a.textContent = label; }
        li.appendChild(a); miList.appendChild(li); miLinks.push(a);
      });
      mi.hidden = false;
    }
  }
  var ticking = false;
  function onScroll() {
    ticking = false;
    var y = window.scrollY, vh = window.innerHeight, H = doc.scrollHeight - vh;
    var frac = H > 0 ? Math.min(1, y / H) : 1;
    if (tokRead) tokRead.textContent = Math.round(totalWords * frac * 4 / 3).toLocaleString('en-US');
    if (cells.length) {
      var probe = y + vh * 0.35;
      h2s.forEach(function (h, i) {
        var start = h.offsetTop, end = (i + 1 < h2s.length ? h2s[i + 1].offsetTop : content.offsetTop + content.offsetHeight);
        var f = Math.max(0, Math.min(1, (probe - start) / Math.max(1, end - start)));
        if (frac > 0.995) f = 1;
        cells[i].firstChild.style.transform = 'scaleX(' + f.toFixed(3) + ')';
        cells[i].classList.toggle('done', f >= 1);
      });
    }
    if (miLinks.length) {
      var active = null;
      miLinks.forEach(function (a) { var el = document.getElementById(a.dataset.id); if (el && el.getBoundingClientRect().top < vh * 0.3) active = a; });
      miLinks.forEach(function (a) { a.classList.toggle('on', a === active); });
    }
    if (frac > 0.7 && !LM.home && here !== '/' && !read[here]) {
      read[here] = Date.now(); store.set('lm-read', read);
      var li = $('.sl-item.is-here'); if (li) li.classList.add('is-read');
    }
  }
  window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
  window.addEventListener('resize', function () { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } });
  onScroll();

  // -------------------------------------------------- command palette
  var pal = $('#palette'), palIn = $('#palInput'), palRes = $('#palResults');
  var index = null, sel = 0, results = [];
  function loadIndex(cb) {
    if (index) return cb();
    var x = new XMLHttpRequest();
    x.open('GET', LM.base + '/assets/data/search.json');
    x.onload = function () { try { index = JSON.parse(x.responseText); } catch (e) { index = []; } cb(); };
    x.onerror = function () { index = []; cb(); };
    x.send();
  }
  function score(item, terms) {
    var hay = (item.t + ' ' + (item.s || '') + ' ' + (item.k || '')).toLowerCase(), s = 0;
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i], p = hay.indexOf(t);
      if (p === -1) return -Infinity;
      s += (item.t.toLowerCase().indexOf(t) !== -1 ? 3 : 1) + (p === 0 ? 1 : 0) + (item.kind === 'page' ? 0.8 : item.kind === 'lab' ? 0.6 : 0);
    }
    return s - hay.length / 400;
  }
  function render() {
    var q = palIn.value.trim().toLowerCase(), terms = q.split(/\s+/).filter(Boolean);
    var pool = index || [];
    results = terms.length ? pool.map(function (it) { return { it: it, s: score(it, terms) }; })
      .filter(function (r) { return r.s > -Infinity; }).sort(function (a, b) { return b.s - a.s; }).slice(0, 12)
      : pool.filter(function (it) { return it.kind === 'page'; }).slice(0, 12).map(function (it) { return { it: it, s: 1 }; });
    var mx = results.length ? Math.max.apply(null, results.map(function (r) { return r.s; })) : 0;
    var Z = results.reduce(function (a, r) { return a + Math.exp(r.s - mx); }, 0) || 1;
    palRes.innerHTML = '';
    if (!results.length) { palRes.innerHTML = '<li class="pal-empty">No match. Try "kv cache", "rope", "lora", "grpo"…</li>'; return; }
    results.forEach(function (r, i) {
      var p = Math.exp(r.s - mx) / Z, it = r.it;
      var li = document.createElement('li'); li.setAttribute('role', 'option'); if (i === sel) li.className = 'sel';
      var a = document.createElement('a'); a.href = LM.base + it.u;
      a.innerHTML = '<span class="pr-n"></span><span class="pr-t"></span><span class="pr-p"></span>';
      a.children[0].textContent = it.n || (it.kind === 'lab' ? '⚗' : '§');
      a.children[1].textContent = it.t;
      if (it.s) { var sm = document.createElement('small'); sm.textContent = it.s; a.children[1].appendChild(sm); }
      a.children[2].innerHTML = 'p=' + p.toFixed(2) + '<i><b style="width:' + (p * 100).toFixed(0) + '%"></b></i>';
      li.appendChild(a); palRes.appendChild(li);
    });
  }
  function openPal() {
    if (!pal) return;
    pal.hidden = false; palIn.value = ''; sel = 0;
    loadIndex(render); setTimeout(function () { palIn.focus(); }, 10);
  }
  function closePal() { if (pal) pal.hidden = true; }
  var sb = $('#searchOpen'); if (sb) sb.addEventListener('click', openPal);
  if (pal) {
    pal.addEventListener('click', function (e) { if (e.target === pal) closePal(); });
    palIn.addEventListener('input', function () { sel = 0; render(); });
    palIn.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { sel = Math.min(sel + 1, results.length - 1); render(); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { sel = Math.max(sel - 1, 0); render(); e.preventDefault(); }
      else if (e.key === 'Enter') { var r = results[sel]; if (r) location.href = LM.base + r.it.u; }
      else if (e.key === 'Escape') closePal();
    });
  }
  document.addEventListener('keydown', function (e) {
    var t = e.target && e.target.tagName;
    var typing = t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || (e.target && e.target.isContentEditable);
    if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) { e.preventDefault(); pal && pal.hidden ? openPal() : closePal(); }
    else if (e.key === '/' && !typing) { e.preventDefault(); openPal(); }
    else if (e.key === 'Escape' && pal && !pal.hidden) closePal();
    else if (e.key === 'Escape' && document.body.classList.contains('stream-open')) setOpen(false);
  });
})();
