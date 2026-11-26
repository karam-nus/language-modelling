/* ==========================================================================
   Language Modelling — figure engine
   <div class="fig" data-fig="NAME" data-…></div> inside a .diagram plate.
   Figures are SVG drawn in the page's ink tokens (var(--blue), var(--verm), …)
   so they follow the paper / night theme. Process figures get a stepper.
   The arithmetic (pipeline schedules, ring all-reduce, masks, …) comes from the
   tested pure functions in lm-labs.js (window.LMPure).
   ========================================================================== */
(function () {
  'use strict';
  var P = window.LMPure || {};
  var NS = 'http://www.w3.org/2000/svg', uid = 0;
  var RANK = ['blue', 'verm', 'teal', 'purple', 'orange', 'green', 'pink', 'yellow'];

  // ------------------------------------------------------------- toolkit
  function E(parent, tag, a, txt) {
    var e = document.createElementNS(NS, tag);
    if (a) Object.keys(a).forEach(function (k) { if (a[k] !== null && a[k] !== undefined) e.setAttribute(k, a[k]); });
    if (txt !== undefined) e.textContent = txt;
    if (parent) parent.appendChild(e);
    return e;
  }
  function H(tag, cls, txt) { var e = document.createElement(tag); if (cls) e.className = cls; if (txt !== undefined) e.textContent = txt; return e; }
  function C(n) { return 'var(--' + n + ')'; }
  function tint(n, p) { return 'color-mix(in srgb, var(--' + n + ') ' + p + '%, var(--card))'; }
  function svg(host, w, h, label) {
    var s = E(null, 'svg', { viewBox: '0 0 ' + w + ' ' + h, class: 'figsvg', role: 'img', 'aria-label': label || '' });
    var id = 'f' + (++uid), defs = E(s, 'defs');
    ['ink', 'ink-3', 'verm', 'blue', 'teal', 'purple', 'orange', 'green', 'pink'].forEach(function (c) {
      var m = E(defs, 'marker', { id: id + '-' + c, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
      E(m, 'path', { d: 'M0 0 L10 5 L0 10 z', style: 'fill:' + C(c) });
    });
    s._id = id;
    host.appendChild(s);
    return s;
  }
  function rect(g, x, y, w, h, o) {
    o = o || {};
    var st = 'fill:' + (o.fill || 'none') + ';stroke:' + (o.stroke === undefined ? C('ink') : o.stroke) + ';stroke-width:' + (o.sw === undefined ? 1.2 : o.sw);
    if (o.dash) st += ';stroke-dasharray:' + o.dash;
    if (o.op !== undefined) st += ';opacity:' + o.op;
    return E(g, 'rect', { x: x, y: y, width: Math.max(0, w), height: Math.max(0, h), rx: o.rx === undefined ? 2 : o.rx, style: st, class: o.cls });
  }
  function text(g, x, y, t, o) {
    o = o || {};
    var st = '';
    if (o.fill) st += 'fill:' + o.fill + ';';
    if (o.size) st += 'font-size:' + o.size + 'px;';
    if (o.weight) st += 'font-weight:' + o.weight + ';';
    if (o.serif) st += 'font-family:var(--font-display);';
    if (o.italic) st += 'font-style:italic;';
    return E(g, 'text', { x: x, y: y, 'text-anchor': o.a || 'middle', class: 'ft', style: st || null }, t);
  }
  function line(g, x1, y1, x2, y2, o) {
    o = o || {};
    var c = o.color || 'ink';
    return E(g, 'line', { x1: x1, y1: y1, x2: x2, y2: y2, style: 'stroke:' + C(c) + ';stroke-width:' + (o.sw || 1.3) + (o.dash ? ';stroke-dasharray:' + o.dash : '') + (o.op !== undefined ? ';opacity:' + o.op : ''), 'marker-end': o.arrow ? 'url(#' + o.svg._id + '-' + c + ')' : null });
  }
  function path(g, d, o) {
    o = o || {};
    var c = o.color || 'ink';
    return E(g, 'path', { d: d, style: 'fill:' + (o.fill || 'none') + ';stroke:' + C(c) + ';stroke-width:' + (o.sw || 1.3) + (o.dash ? ';stroke-dasharray:' + o.dash : '') + (o.op !== undefined ? ';opacity:' + o.op : ''), 'marker-end': o.arrow ? 'url(#' + o.svg._id + '-' + c + ')' : null });
  }
  function group(s) { return E(s, 'g'); }
  function clear(g) { while (g.firstChild) g.removeChild(g.firstChild); }
  // a striped cell showing which ranks have been summed into it
  function sumCell(g, x, y, w, h, mask, N) {
    var ranks = []; for (var r = 0; r < N; r++) if (mask & (1 << r)) ranks.push(r);
    var sw = w / ranks.length;
    ranks.forEach(function (r, i) { rect(g, x + i * sw, y, sw, h, { fill: C(RANK[r]), stroke: 'none', rx: 0 }); });
    rect(g, x, y, w, h, { sw: 1, rx: 1 });
  }

  // a ◀ ▶ ▷ control bar under a figure
  function stepper(host, n, render, label, opts) {
    opts = opts || {};
    var bar = H('div', 'fig-steps'), prev = H('button', 'fs-btn', '◀'), play = H('button', 'fs-btn fs-play', '▷ play'), next = H('button', 'fs-btn', '▶'), lab = H('span', 'fs-label'), dots = H('span', 'fs-dots');
    [prev, play, next].forEach(function (b) { b.type = 'button'; });
    prev.setAttribute('aria-label', 'previous step'); next.setAttribute('aria-label', 'next step');
    lab.setAttribute('aria-live', 'polite');
    for (var k = 0; k < n; k++) { var d = H('i'); d.dataset.k = k; dots.appendChild(d); }
    bar.appendChild(prev); bar.appendChild(play); bar.appendChild(next); bar.appendChild(dots); bar.appendChild(lab);
    host.appendChild(bar);
    var i = 0, timer = null;
    function go(k) {
      i = ((k % n) + n) % n; render(i); lab.innerHTML = label(i);
      Array.prototype.forEach.call(dots.children, function (d, j) { d.className = j === i ? 'on' : j < i ? 'past' : ''; });
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = '▷ play'; } }
    prev.onclick = function () { stop(); go(i - 1); };
    next.onclick = function () { stop(); go(i + 1); };
    dots.onclick = function (e) { if (e.target.dataset.k !== undefined) { stop(); go(+e.target.dataset.k); } };
    play.onclick = function () {
      if (timer) { stop(); return; }
      if (i === n - 1) go(0);
      play.textContent = '❚❚ pause';
      timer = setInterval(function () { if (i === n - 1) { stop(); return; } go(i + 1); }, opts.ms || 1500);
    };
    go(opts.start || 0);
    return { go: go };
  }
  function legend(host, items) {
    var l = H('div', 'legend fig-legend');
    items.forEach(function (it) { var s = H('span'), i = H('i'); i.style.background = it[1]; s.appendChild(i); s.appendChild(document.createTextNode(it[0])); l.appendChild(s); });
    host.appendChild(l);
  }

  var FIGS = {};

  // =====================================================================
  //  ATTENTION
  // =====================================================================

  // How query heads share key/value heads; cache sizes for a 32-head, d_head = 128 layer in BF16.
  FIGS.heads = function (host) {
    var W = 760, Hh = 262, s = svg(host, W, Hh, 'Query heads and the key/value heads they read: MHA, GQA, MQA and MLA');
    var panels = [
      { k: 'MHA', sub: 'one K, V per query head', kv: 8, cache: 16384, note: '32 K + 32 V heads' },
      { k: 'GQA', sub: 'groups share a K, V head', kv: 2, cache: 4096, note: '8 K + 8 V heads' },
      { k: 'MQA', sub: 'one K, V for all heads', kv: 1, cache: 512, note: '1 K + 1 V head' },
      { k: 'MLA', sub: 'cache a latent, expand per head', kv: -1, cache: 1152, note: 'latent 512 + RoPE key 64' }
    ];
    var pick = host.dataset.variant && host.dataset.variant !== 'all' ? panels.filter(function (p) { return p.k.toLowerCase() === host.dataset.variant; }) : panels;
    var pw = 176, gap = (W - pick.length * pw) / Math.max(1, pick.length - 1), maxC = 16384;
    pick.forEach(function (p, pi) {
      var x0 = pick.length === 1 ? (W - pw) / 2 : pi * (pw + gap), g = group(s);
      rect(g, x0, 4, pw, Hh - 8, { fill: C('paper'), stroke: C('rule'), sw: 1 });
      text(g, x0 + 10, 26, p.k, { a: 'start', size: 15, weight: 700, serif: true, fill: C('ink') });
      text(g, x0 + 10, 42, p.sub, { a: 'start', size: 9.5 });
      var hx = function (i) { return x0 + 24 + i * 18.5; };
      text(g, x0 + 10, 70, 'Q', { a: 'start', weight: 700, fill: C('blue') });
      for (var i = 0; i < 8; i++) rect(g, hx(i), 56, 13, 20, { fill: tint('blue', 70), stroke: C('blue') });
      if (p.kv > 0) {
        var per = 8 / p.kv;
        for (var k = 0; k < p.kv; k++) {
          var cx = (hx(k * per) + hx(k * per + per - 1) + 13) / 2, kw = p.kv === 8 ? 13 : 13 + (per - 1) * 3;
          for (var q = k * per; q < (k + 1) * per; q++) line(g, hx(q) + 6.5, 77, cx, 116, { color: 'ink-3', sw: 1 });
          rect(g, cx - kw / 2, 118, kw, 18, { fill: tint('verm', 75), stroke: C('verm') });
          rect(g, cx - kw / 2, 142, kw, 18, { fill: tint('orange', 70), stroke: C('orange') });
        }
        text(g, x0 + 10, 131, 'K', { a: 'start', weight: 700, fill: C('verm') });
        text(g, x0 + 10, 155, 'V', { a: 'start', weight: 700, fill: C('orange') });
      } else {
        var lx = x0 + 36, lw = 92;
        rect(g, lx, 96, lw, 16, { fill: tint('purple', 80), stroke: C('purple') });
        text(g, lx + lw / 2, 108, 'c_KV · 512', { size: 9, weight: 700, fill: '#fff' });
        rect(g, lx + lw + 6, 96, 26, 16, { fill: tint('pink', 75), stroke: C('pink') });
        text(g, lx + lw + 19, 108, 'k_R', { size: 8.5, weight: 700, fill: '#fff' });
        for (i = 0; i < 8; i++) {
          line(g, lx + lw / 2, 113, hx(i) + 6.5, 130, { color: 'purple', sw: 0.8, op: 0.7 });
          rect(g, hx(i), 132, 13, 13, { fill: tint('verm', 30), stroke: C('verm'), dash: '2 2' });
          rect(g, hx(i), 149, 13, 13, { fill: tint('orange', 30), stroke: C('orange'), dash: '2 2' });
        }
        text(g, x0 + 10, 92, 'cached:', { a: 'start', size: 8.5, fill: C('purple') });
        text(g, x0 + 10, 142, 'K', { a: 'start', weight: 700, fill: C('verm') });
        text(g, x0 + 10, 159, 'V', { a: 'start', weight: 700, fill: C('orange') });
        for (i = 0; i < 8; i++) line(g, hx(i) + 6.5, 77, hx(i) + 6.5, 130, { color: 'ink-3', sw: 0.6, dash: '2 3' });
      }
      text(g, x0 + 10, 186, 'KV cache / token / layer', { a: 'start', size: 8.5 });
      var bw = (pw - 20) * Math.sqrt(p.cache / maxC);
      rect(g, x0 + 10, 192, pw - 20, 12, { fill: C('paper-2'), stroke: 'none' });
      rect(g, x0 + 10, 192, bw, 12, { fill: pi === 0 ? C('ink-3') : C('verm'), stroke: 'none' });
      text(g, x0 + 10, 222, p.cache % 1024 === 0 ? (p.cache / 1024) + ' KiB' : p.cache.toLocaleString('en-US') + ' B', { a: 'start', size: 15, weight: 700, serif: true, fill: C('ink') });
      text(g, x0 + pw - 10, 222, pi === 0 && p.k === 'MHA' ? 'baseline' : (p.cache / 16384 * 100).toFixed(p.cache < 1024 ? 1 : 0) + '% of MHA', { a: 'end', size: 9 });
      text(g, x0 + 10, 240, p.note, { a: 'start', size: 8.5 });
    });
    text(s, W / 2, Hh - 1, '', {});
  };

  // Which (query, key) pairs a layer computes, for a short sequence
  var MASKS = {
    causal: ['Causal', 'decoder LMs', {}],
    bidirectional: ['Bidirectional', 'encoders (BERT)', {}],
    prefix: ['Prefix-LM', 'bidirectional prompt', { prefix: 6 }],
    sliding: ['Sliding window', 'W = 5 (Mistral, Gemma local)', { window: 5 }],
    chunked: ['Chunked', 'C = 4 (Llama 4 local layers)', { chunk: 4 }],
    sink: ['Sinks + window', '2 sinks, W = 4 (StreamingLLM)', { sinks: 2, window: 4 }],
    nsa: ['Block-sparse', 'top blocks + window (NSA)', { block: 4, blocks: 1, window: 3 }],
    topk: ['Top-k selected', 'k = 4 per query (DeepSeek DSA)', { topk: 4 }],
    dilated: ['Dilated', 'every 2nd key, W = 4', { window: 4 }]
  };
  function drawMask(g, x0, y0, kind, T, cs, opt) {
    var m = P.maskPattern(kind, T, opt), st = P.maskStats(m);
    var col = { 1: tint('blue', 78), 2: C('verm'), 3: tint('teal', 85) };
    for (var i = 0; i < T; i++) for (var j = 0; j < T; j++) {
      rect(g, x0 + j * cs, y0 + i * cs, cs - 1, cs - 1, { fill: m[i][j] ? col[m[i][j]] : C('paper-2'), stroke: 'none', rx: 0.5 });
    }
    return st;
  }
  FIGS.masks = function (host) {
    var set = (host.dataset.set || 'causal,sliding,chunked,sink,nsa,topk').split(','), T = +(host.dataset.t || 16);
    var per = Math.min(set.length, +(host.dataset.cols || 3)), rows = Math.ceil(set.length / per), pw = 760 / per, cs = Math.min(10, (pw - 40) / T);
    var ph = T * cs + 70, s = svg(host, 760, rows * ph, 'Attention masks: which query–key pairs each pattern computes');
    set.forEach(function (k, idx) {
      var spec = MASKS[k], x0 = (idx % per) * pw + (pw - T * cs) / 2, y0 = Math.floor(idx / per) * ph + 30, g = group(s);
      text(g, x0, y0 - 16, spec[0], { a: 'start', size: 13, weight: 700, serif: true, fill: C('ink') });
      text(g, x0, y0 - 4, spec[1], { a: 'start', size: 8.5 });
      var st = drawMask(g, x0, y0, k, T, cs, spec[2]);
      text(g, x0, y0 + T * cs + 14, 'pairs ' + st.pairs + ' (' + Math.round(st.frac * 100) + '% of causal)', { a: 'start', size: 8.5 });
      text(g, x0, y0 + T * cs + 26, 'last query reads ' + st.lastRow + ' of ' + T + ' keys', { a: 'start', size: 8.5 });
    });
    legend(host, [['attended', tint('blue', 78)], ['local window', tint('teal', 85)], ['sinks / selected', C('verm')], ['skipped', C('paper-2')]]);
  };

  // A fixed-size state vs a growing cache, token by token
  FIGS.linstate = function (host) {
    var W = 760, Hh = 250, s = svg(host, W, Hh, 'Linear attention keeps a fixed d×d state while softmax attention keeps every key and value'), g = group(s);
    var d = 6, T = 10, rnd = P.mulberry32 ? P.mulberry32(9) : Math.random, K = [], V = [], i, j;
    for (i = 0; i < T; i++) { K.push([]); V.push([]); for (j = 0; j < d; j++) { K[i].push(rnd() * 2 - 1); V[i].push(rnd() * 2 - 1); } }
    stepper(host, T, function (t) {
      clear(g);
      text(g, 20, 22, 'Softmax attention: KV cache', { a: 'start', size: 13, weight: 700, serif: true, fill: C('ink') });
      text(g, 20, 37, 'every token appends a key and a value', { a: 'start', size: 9 });
      for (var n = 0; n <= t; n++) for (var r = 0; r < d; r++) {
        rect(g, 20 + n * 15, 50 + r * 13, 12, 11, { fill: tint('verm', 25 + Math.abs(K[n][r]) * 70), stroke: 'none', rx: 1 });
        rect(g, 20 + n * 15, 136 + r * 13, 12, 11, { fill: tint('orange', 25 + Math.abs(V[n][r]) * 70), stroke: 'none', rx: 1 });
      }
      text(g, 8, 90, 'K', { weight: 700, fill: C('verm') }); text(g, 8, 176, 'V', { weight: 700, fill: C('orange') });
      text(g, 20, 238, 'numbers held: 2 × ' + (t + 1) + ' × d = ' + 2 * (t + 1) * d + ' — grows with every token', { a: 'start', size: 9.5, fill: C('ink-2') });
      var x0 = 430;
      text(g, x0, 22, 'Linear attention: state S', { a: 'start', size: 13, weight: 700, serif: true, fill: C('ink') });
      text(g, x0, 37, 'S ← S + v kᵀ  (delta rule: S ← S + β(v − S k) kᵀ)', { a: 'start', size: 9 });
      var S = []; for (r = 0; r < d; r++) { S.push([]); for (var c = 0; c < d; c++) S[r].push(0); }
      for (n = 0; n <= t; n++) for (r = 0; r < d; r++) for (c = 0; c < d; c++) S[r][c] += V[n][r] * K[n][c];
      var mx = 0.001; S.forEach(function (row) { row.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); }); });
      for (r = 0; r < d; r++) for (c = 0; c < d; c++) {
        var v = S[r][c] / mx;
        rect(g, x0 + c * 24, 54 + r * 24, 22, 22, { fill: v >= 0 ? tint('blue', 15 + v * 80) : tint('purple', 15 - v * 80), stroke: 'none', rx: 1 });
      }
      text(g, x0 + 3 * 24, 210, 'd × d — same size at every step', { size: 9 });
      text(g, x0, 238, 'numbers held: d² = ' + d * d + ' — constant; old pairs blur together', { a: 'start', size: 9.5, fill: C('ink-2') });
      // the new token's write
      line(g, 20 + t * 15 + 6, 48, x0 - 14, 80, { svg: s, color: 'ink-3', dash: '3 3', arrow: true });
    }, function (t) { return '<b>token ' + (t + 1) + '</b> — the cache grows by one column; the state is overwritten in place'; }, { ms: 900 });
  };

  // FlashAttention: one query block streams through the key/value blocks with an online softmax
  FIGS.flash = function (host) {
    var W = 760, Hh = 270, s = svg(host, W, Hh, 'FlashAttention processing one query block against successive key/value blocks'), g = group(s);
    var nb = 4, qb = 3;
    stepper(host, qb + 2, function (st) {
      clear(g);
      text(g, 20, 22, 'Score matrix S = QKᵀ (never materialised)', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
      var cs = 42, x0 = 40, y0 = 40;
      for (var i = 0; i < nb; i++) for (var j = 0; j < nb; j++) {
        var masked = j > i, active = i === qb && j === st && st <= qb, done = i === qb && j < Math.min(st, qb + 1);
        rect(g, x0 + j * cs, y0 + i * cs, cs - 3, cs - 3, { fill: masked ? C('paper-2') : active ? C('verm') : done ? tint('verm', 35) : tint('blue', 18), stroke: i === qb ? C('ink') : 'none', sw: i === qb ? 1.2 : 0 });
        if (masked && i === qb) text(g, x0 + j * cs + 19, y0 + i * cs + 24, '—', { size: 9 });
      }
      for (j = 0; j < nb; j++) text(g, x0 + j * cs + 19, y0 - 6, 'K' + j, { size: 9 });
      for (i = 0; i < nb; i++) text(g, x0 - 16, y0 + i * cs + 24, 'Q' + i, { size: 9 });
      text(g, x0 + 2 * cs, y0 + nb * cs + 18, 'causal: blocks above the diagonal are skipped', { size: 8.5 });
      // HBM and SRAM
      var hx = 260;
      rect(g, hx, 36, 480, 80, { fill: C('paper-2'), stroke: C('rule-2') });
      text(g, hx + 10, 52, 'HBM (large, slow)', { a: 'start', size: 10, weight: 700, fill: C('ink-2') });
      ['Q', 'K', 'V', 'O'].forEach(function (m, mi) {
        for (var b = 0; b < nb; b++) {
          var hot = (m === 'Q' && b === qb) || ((m === 'K' || m === 'V') && b === st && st <= qb) || (m === 'O' && b === qb && st === qb + 1);
          rect(g, hx + 14 + mi * 116 + b * 24, 64, 20, 38, { fill: hot ? C(m === 'O' ? 'green' : m === 'Q' ? 'blue' : 'verm') : tint(m === 'O' ? 'green' : m === 'Q' ? 'blue' : 'verm', 22), stroke: 'none' });
        }
        text(g, hx + 14 + mi * 116 + 48, 112, m, { size: 9, weight: 700 });
      });
      rect(g, hx, 140, 480, 112, { fill: tint('yellow', 14), stroke: C('ink') });
      text(g, hx + 10, 157, 'SRAM (on-chip, ~228 KB per SM on H100)', { a: 'start', size: 10, weight: 700, fill: C('ink-2') });
      var lines = [];
      if (st <= qb) {
        lines = ['load Q' + qb + ' (once) and K' + st + ', V' + st,
          'S = Q' + qb + ' K' + st + 'ᵀ' + (st === qb ? '  (apply causal mask in this diagonal block)' : ''),
          'm_new = max(m, rowmax S);  ℓ ← ℓ·e^(m−m_new) + rowsum e^(S−m_new)',
          'O ← O·e^(m−m_new) + e^(S−m_new) V' + st];
      } else {
        lines = ['all key blocks done for Q' + qb, 'O ← O / ℓ   (normalise once at the end)', 'write O' + qb + ' and the log-sum-exp (for the backward pass) to HBM', 'HBM traffic: Q, K, V, O once each — no T×T matrix'];
      }
      lines.forEach(function (l, li) { text(g, hx + 14, 178 + li * 18, l, { a: 'start', size: 10, fill: li === 0 ? C('ink') : C('ink-2') }); });
      if (st <= qb) { line(g, hx + 14 + 116 + st * 24 + 10, 102, hx + 120, 140, { svg: s, color: 'verm', arrow: true, sw: 1.2 }); }
    }, function (st) { return st <= qb ? '<b>block ' + st + '</b> — scores for one tile live only in SRAM; running max and sum rescale earlier work' : '<b>finish</b> — one normalisation, one write'; }, { ms: 1600 });
  };

  // Attention sink vs gated attention (illustrative maps)
  FIGS.gated = function (host) {
    var W = 760, Hh = 200, s = svg(host, W, Hh, 'An attention sink on the first token and how an output gate removes it'), T = 9;
    function mapAt(x0, title, sub, sink) {
      var g = group(s), cs = 14, rnd = P.mulberry32(4);
      text(g, x0, 20, title, { a: 'start', size: 13, weight: 700, serif: true, fill: C('ink') });
      text(g, x0, 34, sub, { a: 'start', size: 9 });
      for (var i = 0; i < T; i++) {
        var row = [], z = 0, j;
        for (j = 0; j <= i; j++) { var v = Math.exp(rnd() * 1.4 + (j === i ? 0.6 : 0)) * (sink && j === 0 && i > 0 ? 9 : 1); row.push(v); z += v; }
        for (j = 0; j < T; j++) rect(g, x0 + j * cs, 46 + i * cs, cs - 1, cs - 1, { fill: j <= i ? tint(sink && j === 0 ? 'verm' : 'blue', 8 + 90 * row[j] / z) : C('paper-2'), stroke: 'none', rx: 0.5 });
      }
      return g;
    }
    mapAt(30, 'Softmax must sum to 1', 'idle heads dump weight on token 0', true);
    var g2 = mapAt(270, 'With an output gate', 'σ(x·W_g) can switch a head off', false);
    var g3 = group(s), gx = 520;
    text(g3, gx, 20, 'Gate per head (σ)', { a: 'start', size: 13, weight: 700, serif: true, fill: C('ink') });
    text(g3, gx, 34, 'query-dependent, often near zero', { a: 'start', size: 9 });
    [0.92, 0.08, 0.61, 0.03, 0.77, 0.15, 0.04, 0.55].forEach(function (v, i) {
      rect(g3, gx, 48 + i * 17, 180, 12, { fill: C('paper-2'), stroke: 'none' });
      rect(g3, gx, 48 + i * 17, 180 * v, 12, { fill: C('teal'), stroke: 'none' });
      text(g3, gx + 186, 58 + i * 17, 'h' + i + ' ' + v.toFixed(2), { a: 'start', size: 8.5 });
    });
    text(s, 30, 190, 'Illustrative maps. Qwen\'s gated attention (2025) multiplies each head\'s output by a sigmoid gate; sinks and loss spikes largely disappear.', { a: 'start', size: 9, fill: C('ink-2') });
    void g2;
  };

  // =====================================================================
  //  DISTRIBUTED TRAINING
  // =====================================================================
  function gpuBox(g, x, y, w, h, r, label) {
    rect(g, x, y, w, h, { fill: tint(RANK[r % 8], 10), stroke: C(RANK[r % 8]), sw: 1.4, rx: 4 });
    text(g, x + 8, y + 15, label || ('GPU ' + r), { a: 'start', size: 10, weight: 700, fill: C(RANK[r % 8]) });
  }

  // Before → after for each collective, 4 ranks × 4 chunks
  FIGS.collectives = function (host) {
    var N = 4, cw = 13, ch = 13, W = 760, Hh = 360, s = svg(host, W, Hh, 'Collectives: broadcast, all-reduce, reduce-scatter, all-gather and all-to-all, before and after');
    var full = (1 << N) - 1;
    var specs = [
      ['All-reduce', 'everyone gets the sum', function (r, c) { return 1 << r; }, function () { return full; }],
      ['Reduce-scatter', 'rank r gets the sum of chunk r', function (r) { return 1 << r; }, function (r, c) { return c === r ? full : 0; }],
      ['All-gather', 'everyone gets every shard', function (r, c) { return c === r ? 1 << r : 0; }, function (r, c) { return 1 << c; }],
      ['All-to-all', 'rank r sends chunk c to rank c', function (r) { return 1 << r; }, function (r, c) { return 1 << c; }],
      ['Broadcast', 'rank 0 sends to all', function (r) { return r === 0 ? 1 : 0; }, function () { return 1; }],
      ['Send / recv', 'pipeline: stage r → r+1', function (r) { return r === 0 ? 1 : 0; }, function (r) { return r === 1 ? 1 : 0; }]
    ];
    specs.forEach(function (sp, i) {
      var x0 = (i % 3) * 253 + 10, y0 = Math.floor(i / 3) * 178 + 6, g = group(s);
      text(g, x0, y0 + 16, sp[0], { a: 'start', size: 13, weight: 700, serif: true, fill: C('ink') });
      text(g, x0, y0 + 30, sp[1], { a: 'start', size: 9 });
      text(g, x0 + 40, y0 + 48, 'before', { size: 8.5 }); text(g, x0 + 160, y0 + 48, 'after', { size: 8.5 });
      for (var r = 0; r < N; r++) {
        var y = y0 + 56 + r * 26;
        text(g, x0, y + 10, 'r' + r, { a: 'start', size: 9, weight: 700, fill: C(RANK[r]) });
        for (var c = 0; c < N; c++) {
          var b = sp[2](r, c), a = sp[3](r, c), bx = x0 + 16 + c * (cw + 2), ax = x0 + 136 + c * (cw + 2);
          if (b) sumCell(g, bx, y, cw, ch, b, N); else rect(g, bx, y, cw, ch, { fill: 'none', stroke: C('rule-2'), dash: '2 2', sw: 0.8 });
          if (a) sumCell(g, ax, y, cw, ch, a, N); else rect(g, ax, y, cw, ch, { fill: 'none', stroke: C('rule-2'), dash: '2 2', sw: 0.8 });
        }
        line(g, x0 + 90, y + 6.5, x0 + 126, y + 6.5, { svg: s, color: 'ink-3', arrow: true, sw: 1 });
      }
    });
    legend(host, [['rank 0', C('blue')], ['rank 1', C('verm')], ['rank 2', C('teal')], ['rank 3', C('purple')], ['striped = summed contributions', 'linear-gradient(90deg,var(--blue) 25%,var(--verm) 25% 50%,var(--teal) 50% 75%,var(--purple) 75%)']]);
  };

  // Ring all-reduce, step by step
  FIGS.ringallreduce = function (host) {
    var N = +(host.dataset.n || 4), W = 760, Hh = 250, s = svg(host, W, Hh, 'Ring all-reduce: reduce-scatter then all-gather, one chunk per step'), g = group(s);
    var total = 2 * (N - 1);
    stepper(host, total + 1, function (step) {
      clear(g);
      var st = P.ringAllReduce(N, step), gw = 150, gx = function (r) { return 30 + r * ((W - 60 - gw) / (N - 1)); };
      for (var r = 0; r < N; r++) {
        gpuBox(g, gx(r), 20, gw, 160, r);
        for (var c = 0; c < N; c++) {
          text(g, gx(r) + 12, 52 + c * 30 + 12, 'chunk ' + c, { a: 'start', size: 8.5 });
          sumCell(g, gx(r) + 62, 52 + c * 30, 76, 18, st.state[r][c], N);
        }
      }
      if (step > 0) st.last.forEach(function (m) {
        var y = 52 + m.chunk * 30 + 9;
        if (m.to === (m.from + 1) % N && m.to > m.from) line(g, gx(m.from) + 140, y, gx(m.to) + 60, y, { svg: s, color: step <= N - 1 ? 'verm' : 'green', arrow: true, sw: 2 });
        else path(g, 'M' + (gx(m.from) + 100) + ' 180 C ' + (gx(m.from) + 100) + ' 236, ' + (gx(m.to) + 100) + ' 236, ' + (gx(m.to) + 100) + ' 184', { svg: s, color: step <= N - 1 ? 'verm' : 'green', arrow: true, sw: 1.6, dash: '4 3' });
      });
      text(g, W / 2, 246, step === 0 ? 'each rank holds its own gradients, split into ' + N + ' chunks' : step <= N - 1 ? 'reduce-scatter step ' + step + ' of ' + (N - 1) + ': send one chunk right, add what arrives' : 'all-gather step ' + (step - N + 1) + ' of ' + (N - 1) + ': pass finished chunks around the ring', { size: 10, fill: C('ink-2') });
    }, function (step) {
      if (step === 0) return '<b>start</b> — 4 ranks, 4 chunks each';
      if (step <= N - 1) return '<b>reduce-scatter ' + step + '/' + (N - 1) + '</b> — every link busy, 1/N of the data per step';
      return step === total ? '<b>done</b> — every rank has the full sum; each sent 2(N−1)/N of the data' : '<b>all-gather ' + (step - N + 1) + '/' + (N - 1) + '</b>';
    }, { ms: 1300 });
  };

  // DDP step: compute then all-reduce, without and with bucketed overlap
  FIGS.ddp = function (host) {
    var W = 760, Hh = 238, s = svg(host, W, Hh, 'A data-parallel training step, with and without overlapping the gradient all-reduce'), g = group(s);
    var u = 30, x0 = 120;
    function row(y, label, segs) {
      text(g, x0 - 10, y + 15, label, { a: 'end', size: 9.5, weight: 600, fill: C('ink-2') });
      segs.forEach(function (sg) { rect(g, x0 + sg[0] * u, y, sg[1] * u - 2, 22, { fill: C(sg[2]), stroke: 'none', rx: 2 }); if (sg[3]) text(g, x0 + sg[0] * u + sg[1] * u / 2, y + 15, sg[3], { size: 9, weight: 700, fill: '#fff' }); });
    }
    text(g, 20, 20, 'Without overlap', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
    row(30, 'GPU 0', [[0, 4, 'blue', 'forward'], [4, 8, 'purple', 'backward'], [16, 2, 'green', 'opt']]);
    row(56, 'GPU 1', [[0, 4, 'blue', 'forward'], [4, 8, 'purple', 'backward'], [16, 2, 'green', 'opt']]);
    row(82, 'network', [[12, 4, 'verm', 'all-reduce all grads']]);
    text(g, 20, 134, 'Bucketed, overlapped (DDP / FSDP default)', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
    var bw = [[4, 2, 'purple', 'L4'], [6, 2, 'purple', 'L3'], [8, 2, 'purple', 'L2'], [10, 2, 'purple', 'L1']];
    row(144, 'GPU 0', [[0, 4, 'blue', 'forward']].concat(bw).concat([[13, 2, 'green', 'opt']]));
    row(170, 'GPU 1', [[0, 4, 'blue', 'forward']].concat(bw).concat([[13, 2, 'green', 'opt']]));
    row(196, 'network', [[6, 1, 'verm', 'b4'], [8, 1, 'verm', 'b3'], [10, 1, 'verm', 'b2'], [12, 1, 'verm', 'b1']]);
    line(g, x0 + 15 * u, 140, x0 + 15 * u, 222, { color: 'green', dash: '3 3' });
    line(g, x0 + 18 * u, 26, x0 + 18 * u, 110, { color: 'ink-3', dash: '3 3' });
    text(g, x0 + 18 * u + 4, 120, 'step ends', { a: 'start', size: 8.5 });
    text(g, 20, 234, 'Green line: the overlapped step ends earlier — gradients of the last layers are reduced while earlier layers are still computing.', { a: 'start', size: 9, fill: C('ink-2') });
  };

  // ZeRO stages: what each of 4 ranks stores
  FIGS.zero = function (host) {
    var N = 4, W = 760, Hh = 248, s = svg(host, W, Hh, 'Per-rank memory under ZeRO stages 0 to 3 with four data-parallel ranks'), g = group(s);
    var stages = [['ZeRO-0 / DDP', [1, 1, 1]], ['ZeRO-1: shard optimizer', [1, 1, 1 / N]], ['ZeRO-2: + gradients', [1, 1 / N, 1 / N]], ['ZeRO-3 / FSDP: + weights', [1 / N, 1 / N, 1 / N]]];
    var bytes = [2, 2, 12], sc = 23, x0 = 168, parts = ['blue', 'teal', 'purple'];
    text(g, x0, 16, 'weights (BF16, 2 B)', { a: 'start', size: 9, fill: C('blue') });
    text(g, x0 + 130, 16, 'gradients (BF16, 2 B)', { a: 'start', size: 9, fill: C('teal') });
    text(g, x0 + 270, 16, 'optimizer: FP32 master + Adam m, v (12 B)', { a: 'start', size: 9, fill: C('purple') });
    stages.forEach(function (st, si) {
      var y = 28 + si * 54, tot = 0;
      text(g, 10, y + 14, st[0], { a: 'start', size: 10.5, weight: 700, fill: C('ink') });
      for (var r = 0; r < N; r++) {
        var x = x0, yy = y + r * 11;
        st[1].forEach(function (f, pi) {
          var w = bytes[pi] * sc;
          if (f === 1) rect(g, x, yy, w - 1, 9, { fill: C(parts[pi]), stroke: 'none', rx: 1 });
          else { rect(g, x, yy, w - 1, 9, { fill: 'none', stroke: C('rule-2'), sw: 0.7, dash: '2 2', rx: 1 }); rect(g, x + r * w / N, yy, w / N - 1, 9, { fill: C(parts[pi]), stroke: 'none', rx: 1 }); }
          x += w + 3;
        });
        if (r === 0) st[1].forEach(function (f, pi) { tot += bytes[pi] * f; });
      }
      text(g, W - 12, y + 26, tot.toFixed(1).replace('.0', '') + ' B/param', { a: 'end', size: 14, weight: 700, serif: true, fill: C('ink') });
      text(g, W - 12, y + 40, 'per rank (N = 4)', { a: 'end', size: 8.5 });
    });
    text(g, 10, 244, 'Each row of 4 bars is one rank. Dashed = held by another rank; fetched (all-gather) or reduced (reduce-scatter) when needed.', { a: 'start', size: 8.5, fill: C('ink-2') });
  };

  // FSDP: what one rank holds as the step moves through the layers
  FIGS.fsdp = function (host) {
    var N = 4, L = 4, W = 760, Hh = 228, s = svg(host, W, Hh, 'FSDP step on one rank: all-gather a layer, compute, free; reduce-scatter gradients in backward'), g = group(s);
    var steps = [];
    for (var l = 0; l < L; l++) steps.push({ ph: 'F', l: l });
    for (l = L - 1; l >= 0; l--) steps.push({ ph: 'B', l: l });
    steps.push({ ph: 'O', l: -1 });
    stepper(host, steps.length, function (k) {
      clear(g);
      var st = steps[k];
      text(g, 20, 20, 'Rank 0 of ' + N + ' — parameters per layer', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
      for (var l = 0; l < L; l++) {
        var y = 34 + l * 34, full = (st.ph === 'F' || st.ph === 'B') && st.l === l;
        text(g, 20, y + 17, 'layer ' + l, { a: 'start', size: 10, fill: full ? C('ink') : C('ink-3'), weight: full ? 700 : 400 });
        for (var r = 0; r < N; r++) {
          var have = r === 0 || full;
          rect(g, 90 + r * 44, y, 40, 24, { fill: have ? tint(RANK[r], r === 0 ? 85 : 55) : 'none', stroke: have ? 'none' : C('rule-2'), dash: have ? null : '3 3' });
        }
        var gradDone = st.ph === 'O' || (st.ph === 'B' && l > st.l);
        if (gradDone) rect(g, 280, y + 4, 40, 16, { fill: C('teal'), stroke: 'none' });
        if (st.ph === 'B' && st.l === l) for (r = 0; r < N; r++) rect(g, 280 + r * 12, y + 4, 10, 16, { fill: tint('teal', 40), stroke: C('teal'), sw: 0.8 });
      }
      text(g, 90 + 88, 30 - 4 + 0, '', {});
      text(g, 168, 182, 'parameter shards (own shard solid)', { size: 8.5 });
      text(g, 300, 182, 'grad shard', { size: 8.5 });
      var msg = {
        F: ['all-gather layer ' + st.l + ' (shards from ranks 1–3)', 'forward through layer ' + st.l, 'free the other shards; prefetch layer ' + (st.l + 1 < L ? st.l + 1 : '—')],
        B: ['all-gather layer ' + st.l + ' again', 'backward: compute full gradients of layer ' + st.l, 'reduce-scatter: keep only this rank\'s gradient shard'],
        O: ['optimizer step on the local shard only', '(FP32 master + Adam m, v for 1/N of the parameters)', 'no communication']
      }[st.ph];
      var bx = 360;
      rect(g, bx, 40, 380, 130, { fill: C('paper'), stroke: C('rule') });
      msg.forEach(function (m, i) { text(g, bx + 14, 68 + i * 26, (i + 1) + '. ' + m, { a: 'start', size: 11, fill: i === 0 ? C('verm-ink') : C('ink-2') }); });
      text(g, 20, 210, 'Peak parameter memory ≈ own shards of every layer + one (or two, with prefetch) gathered layers.', { a: 'start', size: 9, fill: C('ink-2') });
    }, function (k) { var st = steps[k]; return st.ph === 'O' ? '<b>optimizer</b>' : '<b>' + (st.ph === 'F' ? 'forward' : 'backward') + ', layer ' + st.l + '</b>'; }, { ms: 1400 });
  };

  // Megatron tensor parallelism of the MLP (and attention heads), 2 ranks
  FIGS.tp = function (host) {
    var mode = host.dataset.variant || 'mlp';
    var W = 760, Hh = mode === 'mlp' ? 268 : 212, s = svg(host, W, Hh, 'Tensor parallelism: column-parallel then row-parallel matrices on two ranks'), g = group(s);
    function mat(x, y, w, h, color, label, sub, split) {
      if (split === 'col') { rect(g, x, y, w / 2, h, { fill: tint('blue', 60), stroke: C('blue') }); rect(g, x + w / 2, y, w / 2, h, { fill: tint('verm', 60), stroke: C('verm') }); }
      else if (split === 'row') { rect(g, x, y, w, h / 2, { fill: tint('blue', 60), stroke: C('blue') }); rect(g, x, y + h / 2, w, h / 2, { fill: tint('verm', 60), stroke: C('verm') }); }
      else rect(g, x, y, w, h, { fill: color, stroke: C('ink'), sw: 1 });
      text(g, x + w / 2, y - 6, label, { size: 10, weight: 700, fill: C('ink') });
      if (sub) text(g, x + w / 2, y + h + 13, sub, { size: 8.5 });
    }
    if (mode === 'mlp') {
      mat(20, 50, 60, 90, tint('ink-3', 35), 'X', '[T, d] on both', null);
      text(g, 98, 100, '×', { size: 18 });
      mat(116, 50, 120, 90, null, 'W_up', 'split by columns', 'col');
      text(g, 254, 100, '=', { size: 18 });
      rect(g, 272, 50, 60, 90, { fill: tint('blue', 60), stroke: C('blue') }); rect(g, 340, 50, 60, 90, { fill: tint('verm', 60), stroke: C('verm') });
      text(g, 336, 44, 'H = SwiGLU(X W_up)', { size: 10, weight: 700, fill: C('ink') });
      text(g, 302, 153, 'rank 0', { size: 8.5, fill: C('blue') }); text(g, 370, 153, 'rank 1', { size: 8.5, fill: C('verm') });
      text(g, 418, 100, '×', { size: 18 });
      mat(436, 40, 70, 110, null, 'W_down', 'split by rows', 'row');
      text(g, 524, 100, '=', { size: 18 });
      rect(g, 542, 50, 50, 40, { fill: tint('blue', 40), stroke: C('blue'), dash: '3 2' }); rect(g, 542, 96, 50, 40, { fill: tint('verm', 40), stroke: C('verm'), dash: '3 2' });
      text(g, 567, 44, 'partial Y', { size: 10, weight: 700, fill: C('ink') });
      line(g, 598, 93, 640, 93, { svg: s, color: 'verm', arrow: true, sw: 2 });
      text(g, 619, 84, 'all-reduce', { size: 8.5, fill: C('verm-ink') });
      mat(648, 50, 60, 90, tint('ink-3', 35), 'Y', '[T, d] on both', null);
      var notes = ['1. Column split: each rank computes half of the hidden units — no communication.', '2. SwiGLU is element-wise, so it runs on each half independently.', '3. Row split: each rank multiplies its half of H by its half of W_down → a partial sum of Y.', '4. One all-reduce of [T, d] sums the partials. Backward mirrors it: one more all-reduce.'];
      notes.forEach(function (n, i) { text(g, 20, 192 + i * 18, n, { a: 'start', size: 10, fill: C('ink-2') }); });
    } else {
      text(g, 20, 24, 'Attention: whole heads per rank', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
      for (var h = 0; h < 8; h++) {
        var r = h < 4 ? 0 : 1, x = 40 + h * 52 + (r ? 30 : 0);
        rect(g, x, 50, 44, 26, { fill: tint(r ? 'verm' : 'blue', 65), stroke: C(r ? 'verm' : 'blue') });
        text(g, x + 22, 67, 'head ' + h, { size: 9, weight: 700, fill: '#fff' });
        rect(g, x, 88, 44, 18, { fill: tint(r ? 'verm' : 'blue', 30), stroke: 'none' });
        text(g, x + 22, 101, 'K,V', { size: 8.5 });
      }
      text(g, 40 + 104, 128, 'rank 0: Q,K,V columns for heads 0–3', { size: 9, fill: C('blue') });
      text(g, 40 + 4 * 52 + 30 + 104, 128, 'rank 1: heads 4–7', { size: 9, fill: C('verm') });
      text(g, 20, 160, 'Q, K, V projections are column-parallel (by head); each rank runs attention for its own heads with no communication.', { a: 'start', size: 10, fill: C('ink-2') });
      text(g, 20, 180, 'The output projection W_O is row-parallel, so its partial sums need one all-reduce — same pattern as the MLP.', { a: 'start', size: 10, fill: C('ink-2') });
      text(g, 20, 200, 'Per block: 2 all-reduces forward + 2 backward, each of size [B, T, d]. GQA needs KV heads ≥ TP (or KV replication).', { a: 'start', size: 10, fill: C('ink-2') });
    }
  };

  // Sequence parallelism around one transformer block (TP = 2)
  FIGS.sp = function (host) {
    var W = 760, Hh = 236, s = svg(host, W, Hh, 'Sequence parallelism: norms and dropout sharded along the sequence between tensor-parallel regions'), g = group(s);
    var rows = [
      [['LayerNorm', 'seq', '[T/t, d]'], ['all-gather', 'comm'], ['Attention (TP)', 'tp', 'heads/t, all T'], ['reduce-scatter', 'comm'], ['dropout + residual', 'seq', '[T/t, d]']],
      [['LayerNorm', 'seq', '[T/t, d]'], ['all-gather', 'comm'], ['MLP (TP)', 'tp', 'd_ff/t, all T'], ['reduce-scatter', 'comm'], ['dropout + residual', 'seq', '[T/t, d]']]
    ];
    rows.forEach(function (row, ri) {
      var x = 30, y = 20 + ri * 76;
      text(g, 14, y + 30, ri ? '②' : '①', { size: 13, fill: C('ink-3') });
      row.forEach(function (it) {
        var w = it[1] === 'comm' ? 104 : 140;
        if (it[1] === 'comm') {
          line(g, x + 6, y + 26, x + w - 10, y + 26, { svg: s, color: 'verm', arrow: true, sw: 2 });
          text(g, x + w / 2, y + 16, it[0], { size: 9, fill: C('verm-ink') });
        } else {
          var col = it[1] === 'tp' ? 'purple' : 'teal';
          rect(g, x, y, w, 52, { fill: tint(col, 22), stroke: C(col) });
          text(g, x + w / 2, y + 22, it[0], { size: 10, weight: 700, fill: C('ink') });
          text(g, x + w / 2, y + 39, it[2], { size: 8.5 });
        }
        x += w;
      });
    });
    path(g, 'M 718 72 C 750 72, 750 96, 718 96 L 40 96', { svg: s, color: 'ink-3', dash: '3 3', sw: 1 });
    rect(g, 14, 170, 14, 11, { fill: tint('teal', 22), stroke: C('teal') }); text(g, 34, 180, 'sequence-parallel: each rank holds T/t of the tokens', { a: 'start', size: 9.5, fill: C('ink-2') });
    rect(g, 380, 170, 14, 11, { fill: tint('purple', 22), stroke: C('purple') }); text(g, 400, 180, 'tensor-parallel: all T tokens, 1/t of heads or hidden units', { a: 'start', size: 9.5, fill: C('ink-2') });
    text(g, 14, 206, 'An all-reduce is a reduce-scatter plus an all-gather, so communication volume is unchanged — but norm and dropout', { a: 'start', size: 9.5, fill: C('ink-2') });
    text(g, 14, 222, 'activations (and the residual stream between blocks) shrink by the TP degree t.', { a: 'start', size: 9.5, fill: C('ink-2') });
  };

  // Pipeline schedules (Gantt), drawn from the tested simulator
  function drawPipeline(s, g, res, p, opts) {
    clear(g);
    opts = opts || {};
    var W = 760, x0 = 64, span = res.makespan, u = (W - x0 - 12) / span, rh = Math.min(26, 190 / p);
    for (var st = 0; st < p; st++) {
      var y = 14 + st * (rh + 4);
      rect(g, x0, y, span * u, rh, { fill: tint('ink-3', 10), stroke: 'none', rx: 0 });
      text(g, x0 - 8, y + rh / 2 + 3.5, 'stage ' + st, { a: 'end', size: 9, weight: 600, fill: C('ink-2') });
    }
    res.ops.forEach(function (o) {
      var y = 14 + o.s * (rh + 4), col = o.t === 'F' ? 'blue' : o.t === 'B' ? 'verm' : 'teal';
      rect(g, x0 + o.start * u + 0.5, y, o.dur * u - 1, rh, { fill: o.c ? tint(col, 55) : C(col), stroke: 'none', rx: 1.5 });
      if (o.dur * u > 11) text(g, x0 + (o.start + o.dur / 2) * u, y + rh / 2 + 3.5, String(o.mb + 1), { size: Math.min(9, o.dur * u * 0.55), weight: 700, fill: '#fff' });
    });
    var yb = 14 + p * (rh + 4) + 12;
    text(g, x0, yb + 4, 'time →  makespan ' + (+span.toFixed(2)) + ' forward-units · bubble ' + (res.bubble * 100).toFixed(1) + '% · stage-0 activations held: ' + (+res.peakActivations[0].toFixed(2)) + ' micro-batches', { a: 'start', size: 9.5, fill: C('ink-2') });
    return yb + 14;
  }
  FIGS.pipeline = function (host) {
    var kind = host.dataset.variant || '1f1b', p = +(host.dataset.p || 4), m = +(host.dataset.m || 8), v = +(host.dataset.v || 2);
    var res = P.simulatePipeline(kind, p, m, v), Hh = 14 + p * (Math.min(26, 190 / p) + 4) + 30, s = svg(host, 760, Hh, 'Pipeline schedule ' + kind), g = group(s);
    drawPipeline(s, g, res, p);
    legend(host, [['forward', C('blue')], [kind === 'zb' ? 'backward (input grad)' : 'backward', C('verm')]].concat(kind === 'zb' ? [['weight grad (fills bubbles)', C('teal')]] : []).concat(kind === 'interleaved' ? [['lighter = second model chunk', tint('blue', 55)]] : []).concat([['idle (bubble)', tint('ink-3', 10)]]));
  };

  // Expert parallelism: route → dispatch → compute → combine (top-1, 4 ranks)
  FIGS.ep = function (host) {
    var N = 4, W = 760, Hh = 270, s = svg(host, W, Hh, 'Expert parallelism: tokens are sent to the GPU that holds their expert and back'), g = group(s);
    var route = [[2, 0, 1, 3], [0, 2, 2, 1], [3, 1, 0, 0], [1, 3, 2, 3]];   // route[r][i] = expert of token i on rank r
    var phases = ['route', 'dispatch', 'experts', 'combine'];
    stepper(host, 4, function (ph) {
      clear(g);
      var gw = 160, gx = function (r) { return 20 + r * 186; };
      for (var r = 0; r < N; r++) {
        gpuBox(g, gx(r), 16, gw, 206, r);
        rect(g, gx(r) + 10, 172, gw - 20, 40, { fill: tint(RANK[r], 25), stroke: C(RANK[r]), dash: '3 2' });
        text(g, gx(r) + gw / 2, 196, 'expert ' + r + (ph === 2 ? ' — computing' : ''), { size: 10, weight: 700, fill: C(RANK[r]) });
      }
      function tok(x, y, owner, e, hi) {
        rect(g, x, y, 30, 22, { fill: C(RANK[owner]), stroke: hi ? C('ink') : 'none', sw: 1.6 });
        text(g, x + 15, y + 15, 'e' + e, { size: 9, weight: 700, fill: '#fff' });
      }
      if (ph === 0 || ph === 3) {
        for (r = 0; r < N; r++) for (var i = 0; i < 4; i++) tok(gx(r) + 12 + i * 36, 40, r, route[r][i], ph === 3);
        text(g, W / 2, 252, ph === 0 ? 'Each rank\'s router picks an expert (top-1 here) for each of its tokens; colour = the rank that owns the token.' : 'Combine all-to-all: outputs return to their owners, weighted by the router probability and added to the residual.', { size: 9.5, fill: C('ink-2') });
        if (ph === 3) for (r = 0; r < N; r++) line(g, gx(r) + gw / 2, 168, gx(r) + gw / 2, 70, { svg: s, color: 'ink-3', arrow: true, dash: '3 3' });
      } else {
        for (var e = 0; e < N; e++) {
          var k = 0;
          for (r = 0; r < N; r++) for (i = 0; i < 4; i++) if (route[r][i] === e) {
            var xx = gx(e) + 12 + (k % 4) * 36, yy = ph === 1 ? 84 + Math.floor(k / 4) * 28 : 84 + Math.floor(k / 4) * 28;
            tok(g, xx, yy, r, e, false); k++;
            if (ph === 1 && r !== e) line(g, gx(r) + 27 + i * 36, 62, xx + 15, yy, { svg: s, color: RANK[r], sw: 1, op: 0.55, arrow: true });
          }
        }
        text(g, W / 2, 252, ph === 1 ? 'Dispatch all-to-all: every rank sends each token to the rank holding its expert — tokens per expert vary (load balance).' : 'Each expert runs its FFN on the tokens it received, as one batched matmul.', { size: 9.5, fill: C('ink-2') });
      }
    }, function (ph) { return '<b>' + (ph + 1) + '. ' + phases[ph] + '</b>' + (ph === 1 || ph === 3 ? ' — all-to-all across the EP group' : ''); }, { ms: 1700 });
  };

  // Ring attention (context parallelism): KV blocks rotate around the ring
  FIGS.ring = function (host) {
    var N = 4, W = 760, Hh = 262, s = svg(host, W, Hh, 'Ring attention: each GPU keeps its query chunk while key/value chunks rotate'), g = group(s);
    var pos = [[30, 34], [240, 34], [240, 150], [30, 150]], bw = 120, bh = 60;
    stepper(host, N, function (step) {
      clear(g);
      text(g, 30, 20, 'GPUs in a ring (send KV to the next GPU)', { a: 'start', size: 11, weight: 700, serif: true, fill: C('ink') });
      for (var r = 0; r < N; r++) {
        var x = pos[r][0], y = pos[r][1], kv = ((r - step) % N + N) % N, skip = kv > r;
        rect(g, x, y, bw, bh, { fill: tint(RANK[r], 14), stroke: C(RANK[r]), sw: 1.4, rx: 6 });
        text(g, x + 10, y + 16, 'GPU ' + r, { a: 'start', size: 9.5, weight: 700, fill: C(RANK[r]) });
        rect(g, x + 10, y + 26, 44, 22, { fill: C(RANK[r]), stroke: 'none' }); text(g, x + 32, y + 41, 'Q' + r, { size: 9.5, weight: 700, fill: '#fff' });
        rect(g, x + 64, y + 26, 46, 22, { fill: skip ? C('paper-2') : C(RANK[kv]), stroke: skip ? C('rule-2') : 'none', dash: skip ? '2 2' : null });
        text(g, x + 87, y + 41, skip ? 'skip' : 'KV' + kv, { size: 9.5, weight: 700, fill: skip ? C('ink-3') : '#fff' });
      }
      line(g, 154, 64, 236, 64, { svg: s, color: 'ink-3', arrow: true, dash: '4 3' });
      line(g, 300, 98, 300, 146, { svg: s, color: 'ink-3', arrow: true, dash: '4 3' });
      line(g, 236, 180, 154, 180, { svg: s, color: 'ink-3', arrow: true, dash: '4 3' });
      line(g, 90, 146, 90, 98, { svg: s, color: 'ink-3', arrow: true, dash: '4 3' });
      var gx0 = 430, cs = 42;
      text(g, gx0, 20, 'Q chunk × KV chunk (causal)', { a: 'start', size: 11, weight: 700, serif: true, fill: C('ink') });
      for (var qi = 0; qi < N; qi++) for (var kj = 0; kj < N; kj++) {
        var stepOf = ((qi - kj) % N + N) % N, isDone = kj <= qi && stepOf <= step, now = kj <= qi && stepOf === step;
        rect(g, gx0 + kj * cs, 32 + qi * cs, cs - 3, cs - 3, { fill: kj > qi ? C('paper-2') : now ? C('verm') : isDone ? tint('verm', 35) : tint('blue', 12), stroke: 'none' });
      }
      for (kj = 0; kj < N; kj++) text(g, gx0 + kj * cs + 19, 32 + N * cs + 12, 'KV' + kj, { size: 8.5 });
      for (qi = 0; qi < N; qi++) text(g, gx0 - 14, 32 + qi * cs + 24, 'Q' + qi, { size: 8.5 });
      var tx = gx0 + N * cs + 16;
      text(g, tx, 50, 'step ' + (step + 1) + ' of ' + N, { a: 'start', size: 12, weight: 700, fill: C('verm-ink') });
      ['compute with the KV chunk in', 'hand while the next one arrives', '', 'online softmax merges the', 'partial results (Chapter 25)'].forEach(function (l, i) { text(g, tx, 72 + i * 14, l, { a: 'start', size: 9 }); });
      text(g, 30, 236, 'Grey = skipped by the causal mask, so GPU 0 idles while GPU 3 works every step. Assigning each GPU chunks i and 2N−1−i', { a: 'start', size: 9, fill: C('ink-2') });
      text(g, 30, 252, '(zig-zag) gives every GPU the same amount of work.', { a: 'start', size: 9, fill: C('ink-2') });
    }, function (step) { return '<b>step ' + (step + 1) + '</b> — every GPU passes its KV chunk to the next GPU'; }, { ms: 1500 });
  };

  // Ulysses: all-to-all from sequence-sharded to head-sharded
  FIGS.ulysses = function (host) {
    var N = 4, Hd = 8, W = 760, Hh = 214, s = svg(host, W, Hh, 'DeepSpeed Ulysses: all-to-all switches from sequence shards to head shards'), g = group(s);
    function grid(x0, title, byHead) {
      text(g, x0, 20, title, { a: 'start', size: 11.5, weight: 700, serif: true, fill: C('ink') });
      for (var sq = 0; sq < N; sq++) for (var h = 0; h < Hd; h++) {
        var owner = byHead ? Math.floor(h / (Hd / N)) : sq;
        rect(g, x0 + h * 22, 34 + sq * 30, 20, 26, { fill: tint(RANK[owner], 70), stroke: 'none' });
      }
      for (var h2 = 0; h2 < Hd; h2++) text(g, x0 + h2 * 22 + 10, 34 + N * 30 + 12, 'h' + h2, { size: 8 });
      for (sq = 0; sq < N; sq++) text(g, x0 - 8, 34 + sq * 30 + 17, 'T' + sq, { a: 'end', size: 8 });
    }
    grid(40, 'Before: GPU r holds tokens T_r, all heads', false);
    line(g, 236, 94, 300, 94, { svg: s, color: 'verm', arrow: true, sw: 2 }); text(g, 268, 84, 'all-to-all', { size: 9, fill: C('verm-ink') });
    grid(330, 'Attention: GPU r holds all tokens, heads 2r, 2r+1', true);
    line(g, 526, 94, 590, 94, { svg: s, color: 'verm', arrow: true, sw: 2 }); text(g, 558, 84, 'all-to-all', { size: 9, fill: C('verm-ink') });
    text(g, 610, 70, 'back to', { a: 'start', size: 10 }); text(g, 610, 86, 'sequence', { a: 'start', size: 10 }); text(g, 610, 102, 'shards', { a: 'start', size: 10 });
    text(g, 40, 196, 'Colour = GPU. Each GPU runs full-length attention for its heads; the degree is capped by the number of KV heads.', { a: 'start', size: 9, fill: C('ink-2') });
  };

  // Device mesh: which GPUs form each parallel group
  FIGS.mesh = function (host) {
    var dims = { TP: 4, CP: 2, PP: 2, DP: 2 }, order = ['TP', 'CP', 'PP', 'DP'], nG = 32, perNode = 8;
    var W = 760, Hh = 236, s = svg(host, W, Hh, 'A 32-GPU device mesh: tensor, context, pipeline and data parallel groups'), g = group(s);
    var bar = H('div', 'fig-steps fig-tabs'), cur = 'TP';
    function coord(rank) { var c = {}, x = rank; order.forEach(function (d) { c[d] = x % dims[d]; x = Math.floor(x / dims[d]); }); return c; }
    function draw() {
      clear(g);
      var groupsOf = {}, gi = 0;
      for (var r = 0; r < nG; r++) {
        var c = coord(r), key = order.filter(function (d) { return d !== cur; }).map(function (d) { return c[d]; }).join(',');
        if (!(key in groupsOf)) groupsOf[key] = gi++;
      }
      for (var node = 0; node < nG / perNode; node++) {
        var nx = 14 + node * 186;
        rect(g, nx, 22, 178, 150, { fill: C('paper'), stroke: C('ink-3'), dash: '4 3', rx: 6 });
        text(g, nx + 8, 16, 'node ' + node + ' · NVLink inside', { a: 'start', size: 9, fill: C('ink-3') });
        for (var k = 0; k < perNode; k++) {
          var rank = node * perNode + k, cc = coord(rank), key2 = order.filter(function (d) { return d !== cur; }).map(function (d) { return cc[d]; }).join(','), grp = groupsOf[key2];
          var x = nx + 10 + (k % 4) * 41, y = 34 + Math.floor(k / 4) * 66;
          rect(g, x, y, 36, 54, { fill: tint(RANK[grp % 8], 70), stroke: C(RANK[grp % 8]), sw: 1.2, rx: 3 });
          text(g, x + 18, y + 16, String(rank), { size: 10, weight: 700, fill: '#fff' });
          text(g, x + 18, y + 31, cur + ' ' + cc[cur], { size: 8, fill: '#fff' });
          text(g, x + 18, y + 45, 'pp' + cc.PP, { size: 7.5, fill: '#fff' });
        }
      }
      var desc = {
        TP: 'TP = 4: groups of 4 GPUs inside one node split every matrix — all-reduce in every layer, so they must share NVLink.',
        CP: 'CP = 2: pairs split one long sequence — ring/all-to-all exchange of keys and values, also kept inside the node here.',
        PP: 'PP = 2: pairs across two nodes hold different layers — point-to-point activations, fine over InfiniBand.',
        DP: 'DP = 2: replicas (or FSDP shards) of the whole pipeline — one gradient reduction per step, the most latency-tolerant.'
      }[cur];
      text(g, 14, 196, desc, { a: 'start', size: 9.5, fill: C('ink-2') });
      text(g, 14, 214, 'Same colour = same ' + cur + ' group (GPUs that communicate for ' + cur + '). Rank = tp + 4·cp + 8·pp + 16·dp.', { a: 'start', size: 9, fill: C('ink-3') });
      Array.prototype.forEach.call(bar.children, function (b) { b.classList.toggle('on', b.dataset.d === cur); });
    }
    order.forEach(function (d) { var b = H('button', 'fs-btn', d + ' groups'); b.type = 'button'; b.dataset.d = d; b.onclick = function () { cur = d; draw(); }; bar.appendChild(b); });
    host.appendChild(bar);
    draw();
  };

  // =====================================================================
  //  LOOPED TRANSFORMERS
  // =====================================================================
  FIGS.loopstack = function (host) {
    var W = 760, Hh = 360, s = svg(host, W, Hh, 'A standard 12-layer stack, a 3-layer block looped 4 times, and the prelude–recurrent–coda design'), g = group(s);
    var R = 4, shades = ['blue', 'teal', 'purple', 'green', 'orange', 'pink', 'verm', 'yellow', 'red', 'blue', 'teal', 'purple'];
    stepper(host, R + 1, function (it) {
      clear(g);
      // (a) standard
      var bx = 30, bh = 20, top = 46;
      text(g, bx, 20, 'Standard: 12 layers', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
      text(g, bx, 34, '12 sets of weights, depth 12', { a: 'start', size: 9 });
      for (var l = 0; l < 12; l++) rect(g, bx, top + (11 - l) * (bh + 3), 150, bh, { fill: tint(shades[l], 65), stroke: 'none' });
      // (b) looped
      var lx = 230;
      text(g, lx, 20, 'Looped: 3 layers × 4', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
      text(g, lx, 34, '3 sets of weights, depth 12', { a: 'start', size: 9 });
      for (var pass = 0; pass < R; pass++) for (l = 0; l < 3; l++) {
        var idx = pass * 3 + l, y = top + (11 - idx) * (bh + 3), active = it > 0 && pass === it - 1, past = pass < it - 1 || it === 0;
        rect(g, lx, y, 150, bh, { fill: tint(['blue', 'teal', 'purple'][l], active ? 90 : past ? 55 : 18), stroke: active ? C('ink') : 'none', sw: 1.4 });
        if (l === 1) text(g, lx + 160, y + 14, 'pass ' + (pass + 1), { a: 'start', size: 8.5, fill: active ? C('verm-ink') : C('ink-3') });
      }
      path(g, 'M ' + (lx - 8) + ' ' + (top + 9 * (bh + 3)) + ' C ' + (lx - 34) + ' ' + (top + 9 * (bh + 3)) + ', ' + (lx - 34) + ' ' + (top + 11 * (bh + 3) + 12) + ', ' + (lx - 8) + ' ' + (top + 11 * (bh + 3) + 12), { svg: s, color: 'verm', sw: 1.6, arrow: true });
      text(g, lx - 24, top + 10 * (bh + 3) + 10, '↻', { size: 14, fill: C('verm') });
      // (c) prelude / recurrent / coda
      var cx = 480;
      text(g, cx, 20, 'Prelude → recurrent block → coda', { a: 'start', size: 12, weight: 700, serif: true, fill: C('ink') });
      text(g, cx, 34, 'the input is re-injected at every pass', { a: 'start', size: 9 });
      rect(g, cx, 300, 200, 26, { fill: tint('ink-3', 30), stroke: 'none' }); text(g, cx + 100, 317, 'prelude P (embed + a few layers) → e', { size: 9, fill: C('ink') });
      rect(g, cx, 54, 200, 26, { fill: tint('ink-3', 30), stroke: 'none' }); text(g, cx + 100, 71, 'coda C (a few layers + head)', { size: 9, fill: C('ink') });
      for (pass = 0; pass < R; pass++) {
        y = 250 - pass * 46; var on = it > 0 && pass === it - 1, done = it === 0 || pass < it - 1;
        rect(g, cx + 20, y, 160, 30, { fill: tint('purple', on ? 85 : done ? 45 : 15), stroke: on ? C('ink') : 'none', sw: 1.4 });
        text(g, cx + 100, y + 19, 'R:  s' + (pass + 1) + ' = R(s' + pass + ', e)', { size: 9.5, weight: 700, fill: on || done ? '#fff' : C('ink-3') });
        line(g, cx + 205, 312, cx + 182, y + 15, { svg: s, color: 'orange', sw: 0.9, arrow: true, op: on || done ? 1 : 0.35 });
      }
      text(g, cx + 214, 300, 'e', { a: 'start', size: 11, weight: 700, fill: C('orange') });
      text(g, 30, 350, it === 0 ? 'Same depth, a quarter of the parameters. Step through the passes to see one block reused.' : 'Pass ' + it + ' of ' + R + ': the same 3 layers run again on the updated hidden state; compute grows with every pass, parameters do not.', { a: 'start', size: 9.5, fill: C('ink-2') });
    }, function (it) { return it === 0 ? '<b>overview</b>' : '<b>pass ' + it + ' of ' + R + '</b>'; }, { ms: 1300 });
  };

  // ------------------------------------------------------------- labs that draw figures
  var LABS = window.LMLabs || {};
  if (LABS.__frame) {
    var F = LABS.__frame;
    LABS.pipeline = function (root) {
      var f = F.frame(root, 'Pipeline schedule simulator', 'Every stage runs its micro-batches in the order the schedule dictates; an operation starts as soon as its input (forward) or output gradient (backward) is ready. The bubble is the idle area.');
      var kS = F.select(f.controls, { label: 'schedule', options: [['gpipe', 'GPipe (all F, then all B)'], ['1f1b', '1F1B'], ['interleaved', 'Interleaved 1F1B'], ['zb', 'Zero-bubble (split B / W)']], value: '1f1b', onchange: draw });
      var pS = F.slider(f.controls, { label: 'stages p', min: 2, max: 8, value: 4, onchange: draw });
      var mS = F.slider(f.controls, { label: 'micro-batches m', min: 1, max: 6, value: 2, map: function (x) { return x * 4; }, onchange: draw });
      var vS = F.slider(f.controls, { label: 'chunks per stage v (interleaved)', min: 2, max: 4, value: 2, onchange: draw });
      var put = F.readouts(f.out, [{ id: 'b', k: 'bubble', hot: true }, { id: 'f', k: 'formula' }, { id: 'a', k: 'stage-0 activations' }]);
      var s = svg(f.out, 760, 260, 'pipeline schedule'), g = group(s);
      var note = H('div', 'lab-note'); f.out.appendChild(note);
      function draw() {
        var kind = kS.get(), p = pS.get(), m = mS.get(), v = vS.get();
        if (kind === 'interleaved' && m % p) m = Math.ceil(m / p) * p;
        var res = P.simulatePipeline(kind, p, m, v);
        var hgt = drawPipeline(s, g, res, p);
        s.setAttribute('viewBox', '0 0 760 ' + hgt);
        put('b', (res.bubble * 100).toFixed(1) + '%', 'idle share of stage time');
        put('f', kind === 'zb' ? '(p−1)(F+B−W)' : kind === 'interleaved' ? '(p−1)/(v·m+p−1)' : '(p−1)/(m+p−1)', kind === 'zb' ? 'bubble time with W filling the gaps' : (P.bubbleFormula(kind, p, m, v) * 100).toFixed(1) + '% from the formula');
        put('a', (+res.peakActivations[0].toFixed(2)) + ' micro-batches', kind === 'gpipe' ? 'GPipe keeps all m alive' : 'bounded by the pipeline depth');
        note.innerHTML = { gpipe: 'GPipe runs every forward first, so stage 0 holds activations for all m micro-batches at once.', '1f1b': '1F1B has the same bubble as GPipe but starts backward passes early: stage 0 never holds more than p micro-batches of activations.', interleaved: 'Each GPU holds v non-adjacent chunks of layers, so the pipeline fills v times faster — at the cost of v× more point-to-point messages. m is rounded up to a multiple of p.', zb: 'Splitting the backward into input-gradient (B) and weight-gradient (W) work lets W fill the cool-down bubble; DeepSeek\'s DualPipe goes further by feeding the pipeline from both ends.' }[kind];
      }
      draw();
    };
  }

  // ------------------------------------------------------------- boot
  function boot() {
    Array.prototype.forEach.call(document.querySelectorAll('.fig[data-fig]'), function (el) {
      var fn = FIGS[el.dataset.fig];
      if (!fn || el.dataset.ready) return;
      el.dataset.ready = '1';
      try { fn(el); } catch (e) { el.textContent = 'Figure failed to render: ' + (e && e.message); if (window.console) console.error(e); }
    });
  }
  window.LMFigs = { FIGS: FIGS, drawPipeline: drawPipeline, boot: boot };
  if (document.readyState === 'complete') boot(); else document.addEventListener('DOMContentLoaded', boot);
})();
