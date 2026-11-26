/* ==========================================================================
   Language Modelling — interactive labs (vanilla JS, no dependencies)
   A page opts in with  <div class="lab" data-lab="NAME"></div>.
   The arithmetic lives in small pure functions, exported for Node so that
   tools/test_labs.js can check them against numbers quoted in the chapters.
   Accelerator specs come from assets/data/accelerators.json (shared with
   the Chips guide); a built-in fallback keeps labs working offline.
   ========================================================================== */
(function () {
  'use strict';

  // ===================================================== numeric helpers
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) { var u = 0, v = 0; while (u === 0) u = rnd(); while (v === 0) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function softmax(xs, temp) {
    var t = temp === undefined ? 1 : temp, m = -Infinity, i;
    for (i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
    var e = xs.map(function (x) { return x === -Infinity ? 0 : Math.exp((x - m) / t); }), z = sum(e);
    return e.map(function (x) { return x / z; });
  }
  function fmtBytes(b) {
    if (!isFinite(b)) return '—';
    var u = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'], i = 0;
    while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (i === 0 || b >= 100 ? b.toFixed(0) : b >= 10 ? b.toFixed(1) : b.toFixed(2)) + ' ' + u[i];
  }
  function fmtGB(b) { var g = b / 1e9; return (g >= 100 ? g.toFixed(0) : g >= 10 ? g.toFixed(1) : g.toFixed(2)) + ' GB'; }
  function fmtNum(x, d) {
    if (!isFinite(x)) return '—';
    var a = Math.abs(x), s = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
    for (var i = 0; i < s.length; i++) if (a >= s[i][0]) { var v = x / s[i][0]; return (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(d === undefined ? 2 : d)) + s[i][1]; }
    return a >= 100 ? x.toFixed(0) : a >= 1 ? x.toFixed(2) : x.toPrecision(3);
  }
  function fmtTick(v) {
    var a = Math.abs(v), s = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
    for (var i = 0; i < s.length; i++) if (a >= s[i][0]) return String(+(v / s[i][0]).toPrecision(3)) + s[i][1];
    return String(+v.toPrecision(3));
  }
  function fmtSci(x) { if (!isFinite(x) || x === 0) return String(x); var e = Math.floor(Math.log10(Math.abs(x))), m = x / Math.pow(10, e); return m.toFixed(2) + '×10' + String(e).split('').map(function (c) { return '⁰¹²³⁴⁵⁶⁷⁸⁹'['0123456789'.indexOf(c)] || (c === '-' ? '⁻' : c); }).join(''); }
  function pct(x, d) { return (x * 100).toFixed(d === undefined ? 1 : d) + '%'; }

  // ===================================================== 1 · n-gram LM
  var CORPUS = [
    'a language model assigns a probability to every possible next token .',
    'the model reads the tokens and predicts the next token .',
    'the next token is sampled from the distribution over the vocabulary .',
    'attention lets every token read from every earlier token .',
    'the query of a token is compared with the key of every earlier token .',
    'the values of earlier tokens are mixed by the attention weights .',
    'the kv cache stores the keys and values of every earlier token .',
    'the kv cache grows with every token the model generates .',
    'decode is limited by memory bandwidth and not by compute .',
    'the model reads every weight from memory for every token .',
    'a larger batch shares the weight reads across many sequences .',
    'quantization stores the weights in fewer bits .',
    'the weights of a large model do not fit on one gpu .',
    'a mixture of experts sends every token to a few experts .',
    'the router picks the top experts for every token .',
    'the experts hold most of the parameters of the model .',
    'reasoning models think for many tokens before they answer .',
    'the model is trained with reinforcement learning on problems with checkable answers .',
    'the reward is high when the final answer is correct .',
    'a reasoning model can spend more tokens on a hard problem .',
    'the tokenizer splits the text into tokens before the model sees it .',
    'the embedding table turns every token into a vector .',
    'rope rotates the query and the key by their positions .',
    'the transformer is a stack of attention and feed-forward layers .',
    'every layer reads from the residual stream and writes back to it .',
    'the residual stream carries information from layer to layer .',
    'scaling laws predict the loss from the compute and the data .',
    'a smaller model trained on more data is cheaper to serve .',
    'an agent calls tools and reads the results before the next step .',
    'the agent reads the output of every tool before it decides what to do next .',
    'retrieval finds documents and puts them in the context of the model .',
    'the model answers from the documents in the context .',
    'the context window limits how many tokens the model can read .',
    'speculative decoding drafts several tokens and the large model checks them .',
    'the large model accepts most of the draft tokens .',
    'lora trains a small low-rank update and freezes the weights .',
    'distillation trains a small model to imitate a large model .',
    'the small model learns from the distribution of the large model .'
  ].join(' ');
  function tokenizeWords(text) { return (String(text).toLowerCase().match(/[a-z0-9][a-z0-9'’-]*|[.,;:!?]/g) || []); }
  function trainNgram(tokens) {
    var tri = {}, bi = {}, uni = {};
    function inc(t, k, v) { var o = t[k] || (t[k] = {}); o[v] = (o[v] || 0) + 1; }
    for (var i = 0; i < tokens.length; i++) {
      uni[tokens[i]] = (uni[tokens[i]] || 0) + 1;
      if (i >= 1) inc(bi, tokens[i - 1], tokens[i]);
      if (i >= 2) inc(tri, tokens[i - 2] + '\u0001' + tokens[i - 1], tokens[i]);
    }
    return { tri: tri, bi: bi, uni: uni };
  }
  function ngramNext(model, ctx) {
    var a = ctx[ctx.length - 2], b = ctx[ctx.length - 1], src, order;
    if (a !== undefined && model.tri[a + '\u0001' + b]) { src = model.tri[a + '\u0001' + b]; order = 3; }
    else if (b !== undefined && model.bi[b]) { src = model.bi[b]; order = 2; }
    else { src = model.uni; order = 1; }
    var keys = Object.keys(src), tot = 0;
    keys.forEach(function (k) { tot += src[k]; });
    return { order: order, dist: keys.map(function (k) { return { tok: k, p: src[k] / tot }; }).sort(function (x, y) { return y.p - x.p || (x.tok < y.tok ? -1 : 1); }) };
  }
  function reshapeDist(dist, temperature, topK) {
    if (!dist.length) return [];
    if (temperature <= 0) return dist.map(function (d, i) { return { tok: d.tok, p: i === 0 ? 1 : 0 }; });
    var logits = dist.map(function (d) { return Math.log(d.p); });
    var p = softmax(logits, temperature);
    if (topK > 0 && topK < p.length) p = p.map(function (x, i) { return i < topK ? x : 0; });   // dist is sorted
    var z = sum(p);
    return dist.map(function (d, i) { return { tok: d.tok, p: p[i] / z }; });
  }
  function sampleIndex(probs, rnd) { var r = rnd(), c = 0; for (var i = 0; i < probs.length; i++) { c += probs[i]; if (r < c) return i; } return probs.length - 1; }

  // ===================================================== 2 · RoPE
  function ropeFreq(i, d, base) { return Math.pow(base, -2 * i / d); }
  function ropeRotate(vec, pos, base) {
    var d = vec.length, out = new Array(d);
    for (var i = 0; i < d / 2; i++) {
      var th = pos * ropeFreq(i, d, base), c = Math.cos(th), s = Math.sin(th), x = vec[2 * i], y = vec[2 * i + 1];
      out[2 * i] = x * c - y * s; out[2 * i + 1] = x * s + y * c;
    }
    return out;
  }
  function ropeScore(q, k, m, n, base) { return dot(ropeRotate(q, m, base), ropeRotate(k, n, base)); }

  // ===================================================== 3 · parameters
  function countParams(c) {
    var dh = c.dh || c.d / c.H;
    var attn = c.d * c.H * dh + 2 * c.d * c.Hkv * dh + c.H * dh * c.d;
    var norms = 2 * c.d, dense = 3 * c.d * c.dff, moe = c.moe;
    var Lm = moe ? c.L - (moe.dense || 0) : 0, Ld = c.L - Lm;
    var expert = moe ? 3 * c.d * moe.dffe : 0, router = moe ? c.d * moe.E : 0;
    var ffnTot = Ld * dense + (moe ? Lm * ((moe.E + (moe.shared || 0)) * expert + router) : 0);
    var ffnAct = Ld * dense + (moe ? Lm * ((moe.k + (moe.shared || 0)) * expert + router) : 0);
    var emb = c.V * c.d, head = c.tied ? 0 : c.V * c.d, at = c.L * attn, nm = c.L * norms + c.d;
    return { attn: at, ffn: ffnTot, ffnActive: ffnAct, norms: nm, emb: emb, head: head,
      total: at + ffnTot + nm + emb + head, active: at + ffnAct + nm + emb + head, perLayerAttn: attn };
  }
  var MODELS = {
    'llama31-8b':   { name: 'Llama 3.1 8B',   V: 128256, d: 4096, L: 32, H: 32, Hkv: 8, dh: 128, dff: 14336, tied: false },
    'llama31-70b':  { name: 'Llama 3.1 70B',  V: 128256, d: 8192, L: 80, H: 64, Hkv: 8, dh: 128, dff: 28672, tied: false },
    'llama31-405b': { name: 'Llama 3.1 405B', V: 128256, d: 16384, L: 126, H: 128, Hkv: 8, dh: 128, dff: 53248, tied: false },
    'qwen3-8b':     { name: 'Qwen3-8B',       V: 151936, d: 4096, L: 36, H: 32, Hkv: 8, dh: 128, dff: 12288, tied: false },
    'qwen3-0.6b':   { name: 'Qwen3-0.6B',     V: 151936, d: 1024, L: 28, H: 16, Hkv: 8, dh: 128, dff: 3072, tied: true },
    'qwen3-30b-a3b':{ name: 'Qwen3-30B-A3B (MoE)', V: 151936, d: 2048, L: 48, H: 32, Hkv: 4, dh: 128, dff: 6144, tied: false, moe: { E: 128, k: 8, shared: 0, dffe: 768, dense: 0 } },
    'mixtral-8x7b': { name: 'Mixtral 8×7B (MoE)', V: 32000, d: 4096, L: 32, H: 32, Hkv: 8, dh: 128, dff: 14336, tied: false, moe: { E: 8, k: 2, shared: 0, dffe: 14336, dense: 0 } }
  };

  // ===================================================== 5 · KV cache
  function kvUnit(c) {                       // bytes per token per (cached) layer
    if (c.variant === 'mla') return (c.dc + c.dr) * c.bytes;
    var heads = c.variant === 'mha' ? c.H : c.variant === 'mqa' ? 1 : c.Hkv;
    return 2 * heads * c.dh * c.bytes;
  }
  function kvBytes(c, T) {                   // one sequence of T tokens
    var u = kvUnit(c), local = c.window > 0 ? (c.localLayers || 0) : 0, glob = c.L - local;
    return u * (glob * T + local * Math.min(T, c.window || T));
  }
  var KV_PRESETS = {
    'llama31-8b-mha': { name: 'Llama 3.1 8B shape as MHA', variant: 'mha', L: 32, H: 32, Hkv: 32, dh: 128, bytes: 2, window: 0, localLayers: 0 },
    'llama31-8b':   { name: 'Llama 3.1 8B (GQA-8)', variant: 'gqa', L: 32, H: 32, Hkv: 8, dh: 128, bytes: 2, window: 0, localLayers: 0 },
    'llama31-70b':  { name: 'Llama 3.1 70B (GQA-8)', variant: 'gqa', L: 80, H: 64, Hkv: 8, dh: 128, bytes: 2, window: 0, localLayers: 0 },
    'llama31-405b': { name: 'Llama 3.1 405B (GQA-8)', variant: 'gqa', L: 126, H: 128, Hkv: 8, dh: 128, bytes: 2, window: 0, localLayers: 0 },
    'qwen3-30b-a3b':{ name: 'Qwen3-30B-A3B (GQA-4)', variant: 'gqa', L: 48, H: 32, Hkv: 4, dh: 128, bytes: 2, window: 0, localLayers: 0 },
    'qwen3-235b':   { name: 'Qwen3-235B-A22B (GQA-4)', variant: 'gqa', L: 94, H: 64, Hkv: 4, dh: 128, bytes: 2, window: 0, localLayers: 0 },
    'deepseek-v3':  { name: 'DeepSeek-V3 (MLA)', variant: 'mla', L: 61, H: 128, Hkv: 128, dh: 128, dc: 512, dr: 64, bytes: 2, window: 0, localLayers: 0 },
    'gpt-oss-120b': { name: 'gpt-oss-120b (½ windowed)', variant: 'gqa', L: 36, H: 64, Hkv: 8, dh: 64, bytes: 2, window: 128, localLayers: 18 },
    'gemma3-27b':   { name: 'Gemma 3 27B (5:1 local)', variant: 'gqa', L: 62, H: 32, Hkv: 16, dh: 128, bytes: 2, window: 1024, localLayers: 52 }
  };

  // ===================================================== 6 · training compute
  function trainFlops(N, D) { return 6 * N * D; }
  function gpuHours(flops, peakTF, mfu) { return flops / (peakTF * 1e12 * mfu) / 3600; }

  // ===================================================== 7 · fine-tuning memory
  // Activation bytes per layer ≈ 34·s·b·h (16-bit, FlashAttention, no recompute; Korthikanti et al. 2022);
  // with full recomputation only the layer input (2·s·b·h) is stored, plus one layer's 34·s·b·h at a time.
  function finetuneMem(o) {
    var N = o.N, lin = o.N - o.embParams, P = o.loraP || 0, r = {};
    if (o.method === 'full') { r.weights = 2 * N; r.grads = 2 * N; r.optim = 12 * N; }
    else {
      r.weights = o.method === 'qlora' ? lin * 4.127 / 8 + o.embParams * 2 : 2 * N;
      r.weights += 2 * P; r.grads = 2 * P; r.optim = 8 * P;
    }
    var sbh = o.seq * o.batch * o.d;
    r.acts = o.ckpt ? o.L * 2 * sbh + 34 * sbh : o.L * 34 * sbh;
    r.logits = o.fusedCE ? 0 : o.seq * o.batch * o.V * 4;
    r.overhead = o.overhead || 0;               // CUDA context, allocator fragmentation, kernels' workspaces
    r.total = r.weights + r.grads + r.optim + r.acts + r.logits + r.overhead;
    return r;
  }

  // ===================================================== 8 · LoRA
  function loraPerRank(c, targets) {
    var dh = c.dh || c.d / c.H, s = 0, q = c.H * dh, kv = c.Hkv * dh;
    var dims = { q: c.d + q, k: c.d + kv, v: c.d + kv, o: q + c.d, gate: c.d + c.dff, up: c.d + c.dff, down: c.dff + c.d };
    targets.forEach(function (t) { s += dims[t]; });
    return s * c.L;
  }
  function energyCaptured(r, decay, n) {     // power-law spectrum σ_i = i^-decay
    var tot = 0, top = 0;
    for (var i = 1; i <= n; i++) { var e = Math.pow(i, -2 * decay); tot += e; if (i <= r) top += e; }
    return top / tot;
  }

  // ===================================================== 9 · GRPO
  function groupAdvantages(rs) {
    var m = sum(rs) / rs.length, v = 0;
    rs.forEach(function (r) { v += (r - m) * (r - m); });
    var sd = Math.sqrt(v / rs.length);
    return rs.map(function (r) { return sd > 0 ? (r - m) / sd : 0; });
  }
  function zeroGradFraction(p, G) { return Math.pow(p, G) + Math.pow(1 - p, G); }

  // ===================================================== 10 · sampling
  function shapeLogits(logits, o) {
    var n = logits.length, i;
    if (o.T <= 0) { var a = 0; for (i = 1; i < n; i++) if (logits[i] > logits[a]) a = i; return logits.map(function (_, j) { return j === a ? 1 : 0; }); }
    var l = logits.slice(), order = logits.map(function (_, j) { return j; }).sort(function (x, y) { return logits[y] - logits[x]; });
    if (o.topK > 0 && o.topK < n) order.slice(o.topK).forEach(function (j) { l[j] = -Infinity; });
    var p = softmax(l, o.T);
    if (o.minP > 0) { var mx = Math.max.apply(null, p); p = p.map(function (x) { return x >= o.minP * mx ? x : 0; }); }
    if (o.topP < 1) {
      var z0 = sum(p), cum = 0, keep = {};
      for (i = 0; i < n; i++) { var j = order[i]; if (p[j] <= 0) continue; if (cum < o.topP * z0) keep[j] = 1; cum += p[j]; }
      p = p.map(function (x, j) { return keep[j] ? x : 0; });
    }
    var z = sum(p);
    return p.map(function (x) { return x / z; });
  }

  // ===================================================== 11 · speculative decoding
  function specTokens(a, K) { return a >= 1 ? K + 1 : (1 - Math.pow(a, K + 1)) / (1 - a); }
  function specSpeedup(a, K, c) { return specTokens(a, K) / (1 + K * c); }

  // ===================================================== 12 · paged allocation
  function requestStream(n, seed, meanLen, maxLen) {
    var rnd = mulberry32(seed), out = [];
    for (var i = 0; i < n; i++) { var x = Math.exp(Math.log(meanLen) - 0.32 + 0.8 * gauss(rnd)); out.push(clamp(Math.round(x), 16, maxLen)); }
    return out;
  }
  function allocate(lens, capacity, block, reserve) {
    var c = { n: 0, used: 0, held: 0 }, p = { n: 0, used: 0, held: 0 };
    lens.forEach(function (len) {
      if (c.held + reserve <= capacity) { c.n++; c.held += reserve; c.used += len; }
      var need = Math.ceil(len / block) * block;
      if (p.held + need <= capacity) { p.n++; p.held += need; p.used += len; }
    });
    c.waste = c.held ? 1 - c.used / c.held : 0; p.waste = p.held ? 1 - p.used / p.held : 0;
    return { contiguous: c, paged: p };
  }

  // ===================================================== 13 · number formats
  var FORMATS = {
    fp32: { name: 'FP32', e: 8, m: 23, bias: 127, kind: 'ieee' },
    bf16: { name: 'BF16', e: 8, m: 7, bias: 127, kind: 'ieee' },
    fp16: { name: 'FP16', e: 5, m: 10, bias: 15, kind: 'ieee' },
    e5m2: { name: 'FP8 E5M2', e: 5, m: 2, bias: 15, kind: 'ieee' },
    e4m3: { name: 'FP8 E4M3', e: 4, m: 3, bias: 7, kind: 'fn' },
    e3m2: { name: 'FP6 E3M2', e: 3, m: 2, bias: 3, kind: 'finite' },
    e2m3: { name: 'FP6 E2M3', e: 2, m: 3, bias: 1, kind: 'finite' },
    e2m1: { name: 'FP4 E2M1', e: 2, m: 1, bias: 1, kind: 'finite' }
  };
  function fpDecode(f, code) {
    var mMask = (1 << f.m) - 1, eMax = (1 << f.e) - 1;
    var s = Math.floor(code / Math.pow(2, f.e + f.m)) & 1, E = Math.floor(code / Math.pow(2, f.m)) & eMax, M = code & mMask;
    var sign = s ? -1 : 1;
    if (f.kind === 'ieee' && E === eMax) return M === 0 ? sign * Infinity : NaN;
    if (f.kind === 'fn' && E === eMax && M === mMask) return NaN;
    if (E === 0) return sign * Math.pow(2, 1 - f.bias) * (M / Math.pow(2, f.m));
    return sign * Math.pow(2, E - f.bias) * (1 + M / Math.pow(2, f.m));
  }
  var _tab = {};
  function fpTable(f) {                      // non-negative finite values in code order
    if (_tab[f.name]) return _tab[f.name];
    var n = Math.pow(2, f.e + f.m), out = [];
    for (var c = 0; c < n; c++) { var v = fpDecode(f, c); if (isFinite(v)) out.push({ c: c, v: v }); }
    return (_tab[f.name] = out);
  }
  function fpEncode(f, x) {                  // round-to-nearest-even; IEEE kinds overflow to inf, others saturate
    if (f === FORMATS.fp32) {
      var dv = new DataView(new ArrayBuffer(4)); dv.setFloat32(0, x);
      return { code: dv.getUint32(0), value: dv.getFloat32(0) };
    }
    var signBit = Math.pow(2, f.e + f.m), neg = x < 0 || (x === 0 && 1 / x < 0), a = Math.abs(x);
    if (isNaN(x)) return { code: NaN, value: NaN };
    var t = fpTable(f), mx = t[t.length - 1];
    if (f.kind === 'ieee') {
      var ulpTop = mx.v - t[t.length - 2].v;
      if (a >= mx.v + ulpTop / 2) { var infCode = ((1 << f.e) - 1) * Math.pow(2, f.m); return { code: (neg ? signBit : 0) + infCode, value: neg ? -Infinity : Infinity }; }
    } else if (a >= mx.v) return { code: (neg ? signBit : 0) + mx.c, value: neg ? -mx.v : mx.v };
    var lo = 0, hi = t.length - 1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (t[mid].v <= a) lo = mid; else hi = mid; }
    var pick = t[lo];
    if (lo < t.length - 1) {
      var d0 = a - t[lo].v, d1 = t[hi].v - a;
      if (d1 < d0 || (d1 === d0 && (t[hi].c & 1) === 0)) pick = t[hi];
    }
    return { code: (neg ? signBit : 0) + pick.c, value: neg ? -pick.v : pick.v };
  }

  // ===================================================== 14 · quantization
  function quantize(x, bits, mode, group) {
    var out = new Array(x.length), g = group > 0 ? group : x.length;
    for (var s = 0; s < x.length; s += g) {
      var e = Math.min(s + g, x.length), i;
      if (mode === 'asym') {
        var lo = Infinity, hi = -Infinity;
        for (i = s; i < e; i++) { lo = Math.min(lo, x[i]); hi = Math.max(hi, x[i]); }
        var qm = Math.pow(2, bits) - 1, sc = (hi - lo) / qm || 1, z = Math.round(-lo / sc);
        for (i = s; i < e; i++) out[i] = (clamp(Math.round(x[i] / sc) + z, 0, qm) - z) * sc;
      } else {
        var mxa = 0, qmax = Math.pow(2, bits - 1) - 1;
        for (i = s; i < e; i++) mxa = Math.max(mxa, Math.abs(x[i]));
        var sc2 = mxa / qmax || 1;
        for (i = s; i < e; i++) out[i] = clamp(Math.round(x[i] / sc2), -qmax, qmax) * sc2;
      }
    }
    return out;
  }
  function relError(x, y) { var n = 0, d = 0; for (var i = 0; i < x.length; i++) { n += (x[i] - y[i]) * (x[i] - y[i]); d += x[i] * x[i]; } return Math.sqrt(n / d); }
  function weightRow(n, seed, outFrac, outMag) {
    var rnd = mulberry32(seed), w = [];
    for (var i = 0; i < n; i++) { var v = 0.02 * gauss(rnd); if (rnd() < outFrac) v *= outMag; w.push(v); }
    return w;
  }

  // ===================================================== 15/16 · fit, decode ceiling, roofline
  function decodeCeiling(bytesPerToken, bwTBs) { return bwTBs * 1e12 / bytesPerToken; }
  function attainable(intensity, peakTF, bwTBs) { return Math.min(peakTF, intensity * bwTBs); }  // TFLOP/s
  function gemmIntensity(M, K, N, bytes) { return 2 * M * K * N / (bytes * (M * K + K * N + M * N)); }

  // ===================================================== 17 · parallel memory
  function parallelMem(o) {               // bytes per GPU for model state
    var shard = o.N / (o.tp * o.pp), w = 2 * shard, g = 2 * shard, opt = 12 * shard;
    if (o.zero >= 1) opt /= o.dp;
    if (o.zero >= 2) g /= o.dp;
    if (o.zero >= 3) w /= o.dp;
    return { weights: w, grads: g, optim: opt, total: w + g + opt };
  }

  // ===================================================== 18 · serving cost
  function costPerMillion(gpuPerHour, tokPerSec) { return gpuPerHour * 1e6 / (tokPerSec * 3600); }
  function servePoint(o, B) {               // simple roofline model of one decode step
    var tMem = (o.weightBytes + B * o.kvBytesPerSeq) / (o.bw * 1e12 * o.effBw);
    var tCmp = 2 * o.activeN * B / (o.peak * 1e12 * o.effCmp);
    var t = Math.max(tMem, tCmp);
    return { step: t, perStream: 1 / t, total: B / t, bound: tMem >= tCmp ? 'memory' : 'compute' };
  }

  // ===================================================== 19 · scaling
  var CHIN = { E: 1.82, A: 482.01, B: 2085.43, a: 0.3478, b: 0.3658 };   // Besiroglu et al. (2024) refit
  function chinLoss(N, D) { return CHIN.E + CHIN.A / Math.pow(N, CHIN.a) + CHIN.B / Math.pow(D, CHIN.b); }
  function chinOptimal(C) {
    var G = Math.pow(CHIN.a * CHIN.A / (CHIN.b * CHIN.B), 1 / (CHIN.a + CHIN.b));
    var N = G * Math.pow(C / 6, CHIN.b / (CHIN.a + CHIN.b));
    return { N: N, D: C / (6 * N) };
  }

  // ===================================================== 20 · MoE routing
  // A toy router: logit_e(x) = w_e·x + s_e. s_e is a learned affinity with positive feedback (an
  // expert that receives more tokens trains faster and becomes more attractive) — the collapse mechanism.
  // 'aux' adds the gradient of an auxiliary loss that pushes mean gate probabilities towards uniform;
  // 'bias' (DeepSeek-V3) adds a per-expert bias used only for top-k selection, nudged against the load.
  function simulateRouting(o) {
    var rnd = mulberry32(o.seed || 7), dim = 8, E = o.E, k = o.k, T = o.tokens, i, j, e;
    var W = [], s = new Array(E).fill(0), bias = new Array(E).fill(0), hist = [];
    for (e = 0; e < E; e++) { W.push([]); for (j = 0; j < dim; j++) W[e].push(0.5 * gauss(rnd)); }
    var cap = Math.ceil(o.capacity * T * k / E), dropped = 0, final = null, mean = T * k / E;
    for (var step = 0; step < o.steps; step++) {
      var load = new Array(E).fill(0), probSum = new Array(E).fill(0);
      for (i = 0; i < T; i++) {
        var x = []; for (j = 0; j < dim; j++) x.push(gauss(rnd));
        var logit = W.map(function (w, ix) { return dot(w, x) + s[ix]; }), p = softmax(logit);
        for (e = 0; e < E; e++) probSum[e] += p[e];
        var sel = logit.map(function (l, ix) { return { ix: ix, v: l + (o.method === 'bias' ? bias[ix] : 0) }; })
          .sort(function (a, b) { return b.v - a.v; }).slice(0, k);
        sel.forEach(function (c) { if (o.capacity > 0 && load[c.ix] >= cap) { dropped++; return; } load[c.ix]++; });
      }
      for (e = 0; e < E; e++) {
        var f = load[e] / (T * k), P = probSum[e] / T;
        s[e] += o.eta * (f * E - 1);                              // rich get richer
        if (o.method === 'aux') s[e] -= o.alpha * (P * E - 1);    // auxiliary-loss gradient
        if (o.method === 'bias') bias[e] -= o.gamma * Math.sign(load[e] - mean);
      }
      final = load; hist.push(Math.max.apply(null, load) / mean);
    }
    var spread = Math.max.apply(null, s) - Math.min.apply(null, s);
    return { load: final, imbalance: hist, dropped: dropped / (o.steps * T * k), dead: final.filter(function (l) { return l === 0; }).length, distortion: spread };
  }

  // ===================================================== 21 · fixed state vs KV cache
  function recallError(n, d, rule, seed) {
    var rnd = mulberry32(seed || 1), K = [], V = [], i, j;
    for (i = 0; i < n; i++) {
      var k = [], v = [], nn = 0;
      for (j = 0; j < d; j++) { var g = gauss(rnd); k.push(g); nn += g * g; v.push(gauss(rnd)); }
      nn = Math.sqrt(nn); K.push(k.map(function (x) { return x / nn; })); V.push(v);
    }
    var num = 0, den = 0;
    if (rule === 'softmax') {
      for (i = 0; i < n; i++) {
        var sc = K.map(function (kk) { return 20 * dot(K[i], kk); }), p = softmax(sc);
        for (j = 0; j < d; j++) { var r = 0; for (var t = 0; t < n; t++) r += p[t] * V[t][j]; num += (r - V[i][j]) * (r - V[i][j]); den += V[i][j] * V[i][j]; }
      }
      return Math.sqrt(num / den);
    }
    var S = []; for (i = 0; i < d; i++) S.push(new Array(d).fill(0));
    for (i = 0; i < n; i++) {
      var kq = K[i], vq = V[i], pred = S.map(function (row) { return dot(row, kq); });
      for (var a = 0; a < d; a++) { var delta = rule === 'linear' ? vq[a] : vq[a] - pred[a]; for (var b = 0; b < d; b++) S[a][b] += delta * kq[b]; }
    }
    for (i = 0; i < n; i++) { var out = S.map(function (row) { return dot(row, K[i]); }); for (j = 0; j < d; j++) { num += (out[j] - V[i][j]) * (out[j] - V[i][j]); den += V[i][j] * V[i][j]; } }
    return Math.sqrt(num / den);
  }

  // ===================================================== 22 · test-time compute
  function votingCurve(o) {                 // Monte Carlo, seeded: majority vote and best-of-N with a noisy verifier
    var rnd = mulberry32(o.seed || 11), maxN = o.maxN, trials = o.trials || 2000, m = o.wrong || 5;
    var w = []; for (var i = 0; i < m; i++) w.push(Math.pow(i + 1, -o.concentration)); var wz = sum(w); w = w.map(function (x) { return x / wz; });
    var maj = new Array(maxN + 1).fill(0), best = new Array(maxN + 1).fill(0), pass = new Array(maxN + 1).fill(0);
    for (var t = 0; t < trials; t++) {
      var counts = new Array(m + 1).fill(0), bestScore = -Infinity, bestOk = false, any = false;
      for (var n = 1; n <= maxN; n++) {
        var ok = rnd() < o.p, ans = ok ? 0 : 1 + sampleIndex(w, rnd);
        counts[ans]++; any = any || ok;
        var s = (ok ? 1 : 0) + o.noise * gauss(rnd);
        if (s > bestScore) { bestScore = s; bestOk = ok; }
        var top = 0; for (var a = 1; a <= m; a++) if (counts[a] > counts[top]) top = a;
        var tie = 0; for (a = 0; a <= m; a++) if (counts[a] === counts[top]) tie++;
        if (top === 0) maj[n] += 1 / tie;
        if (bestOk) best[n]++;
        if (any) pass[n]++;
      }
    }
    return { maj: maj.map(function (x) { return x / trials; }), best: best.map(function (x) { return x / trials; }), pass: pass.map(function (x) { return x / trials; }) };
  }

  // ===================================================== 23 · BPE
  function bpeTrain(text, nMerges) {
    var words = {}; (String(text).split(/\s+/).filter(Boolean)).forEach(function (w) { words[w] = (words[w] || 0) + 1; });
    var corpus = Object.keys(words).map(function (w) { return { sym: w.split(''), f: words[w] }; });
    var merges = [], trace = [];
    for (var it = 0; it < nMerges; it++) {
      var pairs = {}, order = [];
      corpus.forEach(function (c) { for (var i = 0; i + 1 < c.sym.length; i++) { var key = c.sym[i] + '\u0000' + c.sym[i + 1]; if (!(key in pairs)) { pairs[key] = 0; order.push(key); } pairs[key] += c.f; } });
      if (!order.length) break;
      var bestK = order[0]; order.forEach(function (k) { if (pairs[k] > pairs[bestK]) bestK = k; });
      var ab = bestK.split('\u0000'), a = ab[0], b = ab[1];
      merges.push([a, b]);
      corpus.forEach(function (c) { var out = []; for (var i = 0; i < c.sym.length; i++) { if (i + 1 < c.sym.length && c.sym[i] === a && c.sym[i + 1] === b) { out.push(a + b); i++; } else out.push(c.sym[i]); } c.sym = out; });
      trace.push({ a: a, b: b, count: pairs[bestK], corpus: corpus.map(function (c) { return { sym: c.sym.slice(), f: c.f }; }) });
    }
    return { merges: merges, trace: trace };
  }
  function bpeEncode(word, merges) {
    var s = word.split('');
    merges.forEach(function (m) { var out = []; for (var i = 0; i < s.length; i++) { if (i + 1 < s.length && s[i] === m[0] && s[i + 1] === m[1]) { out.push(m[0] + m[1]); i++; } else out.push(s[i]); } s = out; });
    return s;
  }

  // ===================================================== 24 · attention masks
  // grid[i][j] for query i, key j: 0 masked · 1 attended · 2 always-kept (sink / selected block) · 3 local window
  function maskPattern(kind, T, o) {
    o = o || {}; var g = [], i, j, W = o.window || 4, C = o.chunk || 4, S = o.sinks || 2, P = o.prefix || 4, K = o.topk || 4, b = o.block || 4, nb = o.blocks || 1;
    var rnd = mulberry32(o.seed || 5);
    for (i = 0; i < T; i++) { g.push([]); for (j = 0; j < T; j++) g[i].push(0); }
    for (i = 0; i < T; i++) {
      for (j = 0; j < T; j++) {
        var c = j <= i, v = 0;
        if (kind === 'bidirectional') v = 1;
        else if (kind === 'causal') v = c ? 1 : 0;
        else if (kind === 'prefix') v = (i < P && j < P) || c ? 1 : 0;
        else if (kind === 'sliding') v = c && i - j < W ? 3 : 0;
        else if (kind === 'chunked') v = c && Math.floor(i / C) === Math.floor(j / C) ? 1 : 0;
        else if (kind === 'sink') v = c ? (j < S ? 2 : i - j < W ? 3 : 0) : 0;
        else if (kind === 'dilated') v = c && (i - j) % 2 === 0 && (i - j) / 2 < W ? 1 : 0;
        g[i][j] = v;
      }
      if (kind === 'topk') {                // DeepSeek sparse attention: a cheap indexer picks k keys per query
        var cand = []; for (j = 0; j < i; j++) cand.push([rnd() + (j < 1 ? 0.6 : 0), j]);
        cand.sort(function (a, b2) { return b2[0] - a[0]; });
        g[i][i] = 1; cand.slice(0, Math.max(0, K - 1)).forEach(function (cj) { g[i][cj[1]] = 2; });
      }
      if (kind === 'nsa') {                 // native sparse attention: compressed blocks + top blocks + window
        var qb = Math.floor(i / b), scores = [];
        for (var bb = 0; bb < qb; bb++) scores.push([rnd() + (bb === 0 ? 0.5 : 0), bb]);
        scores.sort(function (a, b2) { return b2[0] - a[0]; });
        var pick = scores.slice(0, nb).map(function (x) { return x[1]; });
        for (j = 0; j <= i; j++) {
          if (i - j < W) g[i][j] = 3;
          else if (pick.indexOf(Math.floor(j / b)) !== -1) g[i][j] = 2;
          else g[i][j] = 0;
        }
      }
    }
    return g;
  }
  function maskStats(g) {
    var T = g.length, pairs = 0, causal = T * (T + 1) / 2, need = 0, i, j;
    for (i = 0; i < T; i++) for (j = 0; j < T; j++) if (g[i][j]) pairs++;
    // keys the last query still reads (a lower bound on what the cache must hold at the end)
    for (j = 0; j < T; j++) if (g[T - 1][j]) need++;
    return { pairs: pairs, causalPairs: causal, frac: pairs / causal, lastRow: need };
  }

  // ===================================================== 25 · pipeline schedules
  // Times in units of one forward of one full stage; a backward costs 2 (or 1 + 1 when split into B and W).
  function pipelineOrder(kind, p, m, v) {
    var orders = [], s, k;
    for (s = 0; s < p; s++) {
      var o = [];
      if (kind === 'gpipe') {
        for (k = 0; k < m; k++) o.push({ t: 'F', mb: k, c: 0 });
        for (k = 0; k < m; k++) o.push({ t: 'B', mb: k, c: 0 });
      } else if (kind === 'interleaved') {
        var total = m * v, fwd = function (q) { return { t: 'F', c: Math.floor(q / p) % v, mb: Math.floor(q / (p * v)) * p + q % p }; },
          bwd = function (q) { return { t: 'B', c: v - 1 - Math.floor(q / p) % v, mb: Math.floor(q / (p * v)) * p + q % p }; };
        var warm = Math.min((p - s - 1) * 2 + (v - 1) * p, total);
        for (k = 0; k < warm; k++) o.push(fwd(k));
        for (k = 0; k < total - warm; k++) { o.push(fwd(warm + k)); o.push(bwd(k)); }
        for (k = total - warm; k < total; k++) o.push(bwd(k));
      } else {                               // 1f1b and zb share the forward/backward order
        var w1 = Math.min(p - s - 1, m);
        for (k = 0; k < w1; k++) o.push({ t: 'F', mb: k, c: 0 });
        for (k = 0; k < m - w1; k++) { o.push({ t: 'F', mb: w1 + k, c: 0 }); o.push({ t: 'B', mb: k, c: 0 }); }
        for (k = m - w1; k < m; k++) o.push({ t: 'B', mb: k, c: 0 });
      }
      orders.push(o);
    }
    return orders;
  }
  function simulatePipeline(kind, p, m, v) {
    v = kind === 'interleaved' ? (v || 2) : 1;
    var split = kind === 'zb', orders = pipelineOrder(kind, p, m, v), V = p * v;
    // integer clock: forward of one chunk = 1 tick; full backward = 2 ticks (or B 1 + W 1)
    var dur = { F: 1, B: split ? 1 : 2, W: 1 };
    var done = {}, key = function (t, mb, vs) { return t + mb + ':' + vs; };
    var ptr = orders.map(function () { return 0; }), busyUntil = orders.map(function () { return 0; });
    var pendingW = orders.map(function () { return []; }), ops = [], inflight = orders.map(function () { return 0; }), peak = orders.map(function () { return 0; });
    var remaining = orders.reduce(function (a, o) { return a + o.length * (split ? 1 : 1); }, 0) + (split ? p * m * v : 0);
    for (var t = 0; remaining > 0 && t < 100000; t++) {
      for (var s = 0; s < p; s++) {
        if (busyUntil[s] > t) continue;
        var next = orders[s][ptr[s]], run = null;
        if (next) {
          var vs = next.c * p + s, ready;
          if (next.t === 'F') ready = vs === 0 || (done[key('F', next.mb, vs - 1)] !== undefined && done[key('F', next.mb, vs - 1)] <= t);
          else ready = (vs === V - 1 ? done[key('F', next.mb, vs)] !== undefined && done[key('F', next.mb, vs)] <= t
                                     : done[key('B', next.mb, vs + 1)] !== undefined && done[key('B', next.mb, vs + 1)] <= t);
          if (ready) { run = { t: next.t, mb: next.mb, c: next.c, vs: vs }; ptr[s]++; }
        }
        if (!run && pendingW[s].length && (split)) run = pendingW[s].shift();
        if (!run) continue;
        var d = dur[run.t], end = t + d;
        busyUntil[s] = end; done[key(run.t, run.mb, run.vs)] = end; remaining--;
        if (run.t === 'F') { inflight[s] += 1 / v; peak[s] = Math.max(peak[s], inflight[s]); }
        if (run.t === 'B') { inflight[s] -= 1 / v; if (split) pendingW[s].push({ t: 'W', mb: run.mb, c: run.c, vs: run.vs }); }
        ops.push({ s: s, t: run.t, mb: run.mb, c: run.c, start: t / v, dur: d / v });
      }
    }
    var span = Math.max.apply(null, ops.map(function (o) { return o.start + o.dur; }));
    var work = m * 3;                         // per device, in full-stage units
    return { ops: ops, makespan: span, bubble: 1 - work / span, peakActivations: peak };
  }
  function bubbleFormula(kind, p, m, v) {
    if (kind === 'interleaved') return (p - 1) / (v * m + p - 1);
    return (p - 1) / (m + p - 1);
  }

  // ===================================================== 26 · collectives
  // Ring all-reduce over N ranks: chunks as contributor sets (bitmasks). Steps 0..N-2 reduce-scatter, N-1..2N-3 all-gather.
  function ringAllReduce(N, steps) {
    var st = [], i, c;
    for (i = 0; i < N; i++) { st.push([]); for (c = 0; c < N; c++) st[i].push(1 << i); }
    var sends = [];
    for (var s = 0; s < steps && s < 2 * (N - 1); s++) {
      var msgs = [];
      for (i = 0; i < N; i++) {
        var chunk = s < N - 1 ? ((i - s) % N + N) % N : ((i + 1 - (s - (N - 1))) % N + N) % N;
        msgs.push({ from: i, to: (i + 1) % N, chunk: chunk, val: st[i][chunk] });
      }
      msgs.forEach(function (mm) { st[mm.to][mm.chunk] = s < N - 1 ? (st[mm.to][mm.chunk] | mm.val) : mm.val; });
      sends = msgs;
    }
    return { state: st, last: sends };
  }
  function popcount(x) { var n = 0; while (x) { n += x & 1; x >>>= 1; } return n; }
  function collectiveTime(kind, N, bytes, bwGBs, latencyUs) {   // ring algorithms; bw = per-GPU one-way bandwidth
    var f = { allreduce: 2 * (N - 1) / N, allgather: (N - 1) / N, reducescatter: (N - 1) / N, alltoall: (N - 1) / N, broadcast: 1 }[kind];
    var steps = { allreduce: 2 * (N - 1), allgather: N - 1, reducescatter: N - 1, alltoall: N - 1, broadcast: N - 1 }[kind];
    return { bytesPerGpu: f * bytes, seconds: f * bytes / (bwGBs * 1e9) + steps * (latencyUs || 0) * 1e-6 };
  }

  // ===================================================== 27 · looped depth
  // A toy recurrent block h ← c·tanh(W h) + e per token; c is the token's "difficulty" (contraction rate).
  // A token exits when its update is small relative to its state, or at rMax.
  function loopSim(o) {
    var rnd = mulberry32(o.seed || 3), d = o.d || 16, n = o.tokens || 12, i, j, k;
    // orthogonal W (Gram–Schmidt): spectral norm 1, so h ← c·tanh(W h) + e contracts at rate ≤ c
    var W = [];
    for (i = 0; i < d; i++) {
      var v = []; for (j = 0; j < d; j++) v.push(gauss(rnd));
      W.forEach(function (u) { var pr = dot(u, v); for (var q = 0; q < d; q++) v[q] -= pr * u[q]; });
      var nv = Math.sqrt(dot(v, v)); W.push(v.map(function (x) { return x / nv; }));
    }
    var e = []; for (i = 0; i < d; i++) e.push(gauss(rnd));   // the same injected input, so only c (difficulty) differs
    var out = [];
    for (k = 0; k < n; k++) {
      var c = o.cMin + (o.cMax - o.cMin) * (n === 1 ? 0.5 : k / (n - 1)), h = [];
      for (i = 0; i < d; i++) h.push(0);
      var res = [], loops = o.rMax;
      for (var r = 1; r <= o.rMax; r++) {
        var nh = W.map(function (row, ii) { return c * Math.tanh(dot(row, h)) + e[ii]; });
        var num = 0, den = 0; for (i = 0; i < d; i++) { num += (nh[i] - h[i]) * (nh[i] - h[i]); den += nh[i] * nh[i]; }
        var rel = Math.sqrt(num / den); res.push(rel); h = nh;
        if (r >= (o.rMin || 1) && rel < o.eps) { loops = r; break; }
      }
      out.push({ c: c, loops: loops, residuals: res });
    }
    var mean = sum(out.map(function (x) { return x.loops; })) / n;
    return { tokens: out, meanLoops: mean, computeFraction: mean / o.rMax };
  }

  var PURE = {
    mulberry32: mulberry32, softmax: softmax, fmtBytes: fmtBytes,
    tokenizeWords: tokenizeWords, trainNgram: trainNgram, ngramNext: ngramNext, reshapeDist: reshapeDist, CORPUS: CORPUS,
    ropeFreq: ropeFreq, ropeRotate: ropeRotate, ropeScore: ropeScore,
    countParams: countParams, MODELS: MODELS,
    kvBytes: kvBytes, kvUnit: kvUnit, KV_PRESETS: KV_PRESETS,
    trainFlops: trainFlops, gpuHours: gpuHours, finetuneMem: finetuneMem,
    loraPerRank: loraPerRank, energyCaptured: energyCaptured,
    groupAdvantages: groupAdvantages, zeroGradFraction: zeroGradFraction,
    shapeLogits: shapeLogits, specTokens: specTokens, specSpeedup: specSpeedup,
    requestStream: requestStream, allocate: allocate,
    FORMATS: FORMATS, fpDecode: fpDecode, fpEncode: fpEncode,
    quantize: quantize, relError: relError, weightRow: weightRow,
    decodeCeiling: decodeCeiling, attainable: attainable, gemmIntensity: gemmIntensity,
    parallelMem: parallelMem, costPerMillion: costPerMillion, servePoint: servePoint,
    chinLoss: chinLoss, chinOptimal: chinOptimal, simulateRouting: simulateRouting,
    recallError: recallError, votingCurve: votingCurve, bpeTrain: bpeTrain, bpeEncode: bpeEncode,
    maskPattern: maskPattern, maskStats: maskStats, pipelineOrder: pipelineOrder, simulatePipeline: simulatePipeline, bubbleFormula: bubbleFormula,
    ringAllReduce: ringAllReduce, popcount: popcount, collectiveTime: collectiveTime, loopSim: loopSim, gauss: gauss
  };
  if (typeof module !== 'undefined' && module.exports) { module.exports = PURE; }
  if (typeof window !== 'undefined') window.LMPure = PURE;
  if (typeof document === 'undefined') return;

  // =====================================================================
  //                               BROWSER UI
  // =====================================================================
  var BASE = (window.LM && window.LM.base) || '';
  function h(tag, attrs) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'text') e.textContent = attrs[k];
      else if (k === 'html') e.innerHTML = attrs[k];
      else if (k === 'class') e.className = attrs[k];
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    });
    for (var i = 2; i < arguments.length; i++) { var c = arguments[i]; if (c == null) continue; e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); }
    return e;
  }
  function frame(root, title, desc, single) {
    var meta = (window.LM_LABS || {})[root.dataset.lab] || {};
    root.innerHTML = '';
    root.id = root.id || ('lab-' + root.dataset.lab);
    var head = h('div', { class: 'lab-head' }, h('span', { class: 'lab-badge', text: 'Lab' }), h('span', { class: 'lab-title', text: title }), h('span', { class: 'lab-live', text: 'live' }));
    root.appendChild(head);
    if (desc) root.appendChild(h('p', { class: 'lab-desc', html: desc }));
    var controls = h('div', { class: 'lab-controls' }), out = h('div', { class: 'lab-out' });
    root.appendChild(h('div', { class: 'lab-body' + (single ? ' single' : '') }, controls, out));
    return { controls: controls, out: out, meta: meta };
  }
  function slider(parent, o) {
    var val = h('span', { class: 'ctl-val' });
    var inp = h('input', { type: 'range', min: o.min, max: o.max, step: o.step || 1, value: o.value, 'aria-label': o.label });
    var c = h('label', { class: 'ctl' }, h('span', { class: 'ctl-label' }, labelText(o.label), val), inp);
    parent.appendChild(c);
    var get = function () { var v = parseFloat(inp.value); return o.map ? o.map(v) : v; };
    var show = function () { val.textContent = o.fmt ? o.fmt(get()) : String(get()); };
    inp.addEventListener('input', function () { show(); o.onchange && o.onchange(); });
    show();
    return { get: get, el: inp, set: function (v) { inp.value = v; show(); } };
  }
  function labelText(t) {                   // keep Greek symbols lower-case inside upper-case labels
    var sp = h('span'); String(t).split(/([\u0370-\u03ff]+)/).forEach(function (part, i) { sp.appendChild(i % 2 ? h('span', { class: 'nt', text: part }) : document.createTextNode(part)); });
    return sp;
  }
  function select(parent, o) {
    var s = h('select', { 'aria-label': o.label });
    o.options.forEach(function (opt) { s.appendChild(h('option', { value: opt[0], text: opt[1] })); });
    if (o.value !== undefined) s.value = o.value;
    parent.appendChild(h('label', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', { text: o.label })), s));
    s.addEventListener('change', function () { o.onchange && o.onchange(); });
    return { get: function () { return s.value; }, el: s };
  }
  function check(parent, o) {
    var i = h('input', { type: 'checkbox' }); i.checked = !!o.value;
    parent.appendChild(h('label', { class: 'chk' }, i, h('span', { text: o.label })));
    i.addEventListener('change', function () { o.onchange && o.onchange(); });
    return { get: function () { return i.checked; } };
  }
  function readouts(parent, items) {
    var box = h('div', { class: 'readouts' }), map = {};
    items.forEach(function (it) {
      var v = h('span', { class: 'r-v' }), s = h('span', { class: 'r-s' });
      var r = h('div', { class: 'readout' + (it.hot ? ' hot' : '') }, h('span', { class: 'r-k', text: it.k }), v, s);
      box.appendChild(r); map[it.id] = { v: v, s: s, r: r };
    });
    parent.appendChild(box);
    return function (id, value, sub) { if (!map[id]) return; map[id].v.textContent = value; map[id].s.textContent = sub || ''; };
  }
  function barList(parent) {
    var box = h('div', { class: 'bars' }); parent.appendChild(box);
    return function (rows) {
      box.innerHTML = '';
      rows.forEach(function (r) {
        var t = h('div', { class: 'b-t' }); var bar = h('i'); bar.style.width = (clamp(r.w, 0, 1) * 100).toFixed(1) + '%';
        if (r.color) bar.style.background = r.color; t.appendChild(bar);
        if (r.w2 !== undefined) { var b2 = h('i'); b2.style.width = (clamp(r.w2, 0, 1) * 100).toFixed(1) + '%'; t.appendChild(b2); }
        box.appendChild(h('div', { class: 'bar-row' + (r.cls ? ' ' + r.cls : '') }, h('span', { class: 'b-k', text: r.k, title: r.k }), t, h('span', { class: 'b-v', text: r.v })));
      });
    };
  }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs, text) { var e = document.createElementNS(SVGNS, tag); Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); }); if (text !== undefined) e.textContent = text; return e; }
  function chart(parent, height) {
    var svg = sv('svg', { class: 'chart', viewBox: '0 0 640 ' + (height || 260), role: 'img' }); parent.appendChild(svg);
    var leg = h('div', { class: 'legend' }); parent.appendChild(leg);
    var COL = { 's-a': 'verm', 's-b': 'blue', 's-c': 'teal', 's-d': 'purple' };
    return function (cfg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      leg.innerHTML = ''; leg.style.display = cfg.legend ? '' : 'none';
      (cfg.series || []).forEach(function (s) { if (!cfg.legend || !s.label) return; var i = h('i'); i.style.background = 'var(--' + COL[s.cls.split(' ')[0]] + ')'; if (/dash/.test(s.cls)) i.style.opacity = '.55'; leg.appendChild(h('span', {}, i, s.label)); });
      var W = 640, H = height || 260, L = 52, R = 14, T = 12, B = 34;
      var tx = function (v) { var a = cfg.x.log ? Math.log10(v) : v, lo = cfg.x.log ? Math.log10(cfg.x.min) : cfg.x.min, hi = cfg.x.log ? Math.log10(cfg.x.max) : cfg.x.max; return L + (a - lo) / (hi - lo) * (W - L - R); };
      var ty = function (v) { var a = cfg.y.log ? Math.log10(v) : v, lo = cfg.y.log ? Math.log10(cfg.y.min) : cfg.y.min, hi = cfg.y.log ? Math.log10(cfg.y.max) : cfg.y.max; return H - B - (a - lo) / (hi - lo) * (H - T - B); };
      function ticks(ax) { if (ax.ticks) return ax.ticks; var out = []; if (ax.log) { for (var e = Math.ceil(Math.log10(ax.min)); e <= Math.floor(Math.log10(ax.max)); e++) out.push(Math.pow(10, e)); } else { var st = (ax.max - ax.min) / 5; for (var v = ax.min; v <= ax.max + 1e-9; v += st) out.push(v); } return out; }
      ticks(cfg.y).forEach(function (v) { svg.appendChild(sv('line', { x1: L, x2: W - R, y1: ty(v), y2: ty(v), class: 'grid' })); svg.appendChild(sv('text', { x: L - 6, y: ty(v) + 3.5, 'text-anchor': 'end' }, (cfg.y.fmt || fmtTick)(v))); });
      ticks(cfg.x).forEach(function (v) { svg.appendChild(sv('line', { x1: tx(v), x2: tx(v), y1: T, y2: H - B, class: 'grid' })); svg.appendChild(sv('text', { x: tx(v), y: H - B + 15, 'text-anchor': 'middle' }, (cfg.x.fmt || fmtTick)(v))); });
      svg.appendChild(sv('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, class: 'axis' }));
      svg.appendChild(sv('line', { x1: L, x2: L, y1: T, y2: H - B, class: 'axis' }));
      if (cfg.x.label) svg.appendChild(sv('text', { x: (L + W - R) / 2, y: H - 4, 'text-anchor': 'middle', class: 'lbl' }, cfg.x.label));
      if (cfg.y.label) svg.appendChild(sv('text', { x: 12, y: (T + H - B) / 2, 'text-anchor': 'middle', class: 'lbl', transform: 'rotate(-90 12 ' + (T + H - B) / 2 + ')' }, cfg.y.label));
      (cfg.areas || []).forEach(function (a) { var d = a.pts.map(function (p, i) { return (i ? 'L' : 'M') + tx(p[0]).toFixed(1) + ' ' + ty(p[1]).toFixed(1); }).join(' '); svg.appendChild(sv('path', { d: d + ' L' + tx(a.pts[a.pts.length - 1][0]) + ' ' + (H - B) + ' L' + tx(a.pts[0][0]) + ' ' + (H - B) + 'Z', fill: a.fill, opacity: a.opacity || 0.15 })); });
      (cfg.series || []).forEach(function (s) {
        var pts = s.pts.filter(function (p) { return isFinite(p[0]) && isFinite(p[1]) && (!cfg.y.log || p[1] > 0); });
        if (!pts.length) return;
        var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + tx(p[0]).toFixed(1) + ' ' + ty(clamp(p[1], cfg.y.min, cfg.y.max)).toFixed(1); }).join(' ');
        svg.appendChild(sv('path', { d: d, class: s.cls + (s.dash ? ' dash' : '') }));
        if (s.label && !cfg.legend) { var lp = pts[pts.length - 1]; svg.appendChild(sv('text', { x: tx(lp[0]) - 4, y: ty(clamp(lp[1], cfg.y.min, cfg.y.max)) - 6, 'text-anchor': 'end', class: 'lbl' }, s.label)); }
      });
      (cfg.hlines || []).forEach(function (v) { svg.appendChild(sv('line', { x1: L, x2: W - R, y1: ty(v.y), y2: ty(v.y), class: 's-d dash' })); if (v.label) svg.appendChild(sv('text', { x: W - R - 4, y: ty(v.y) - 4, 'text-anchor': 'end', class: 'lbl' }, v.label)); });
      (cfg.vlines || []).forEach(function (v) { svg.appendChild(sv('line', { x1: tx(v.x), x2: tx(v.x), y1: T, y2: H - B, class: 's-d dash' })); if (v.label) svg.appendChild(sv('text', { x: tx(v.x) + 4, y: T + 10, class: 'lbl' }, v.label)); });
      (cfg.points || []).forEach(function (p) {
        var cx = tx(p.x), cy = ty(clamp(p.y, cfg.y.min, cfg.y.max));
        svg.appendChild(sv('circle', { cx: cx, cy: cy, r: p.r || 5, fill: p.fill || 'var(--verm)', stroke: 'var(--paper)', 'stroke-width': 1.5 }));
        if (p.label) svg.appendChild(sv('text', { x: cx + (p.dx || 8), y: cy + (p.dy || 4), class: 'ink', 'text-anchor': p.anchor || 'start' }, p.label));
      });
    };
  }

  // shared accelerator data (from the Chips guide) with an offline fallback
  var ACCEL_FALLBACK = [
    { id: 'rtx4090', name: 'GeForce RTX 4090', mem_gb: 24, mem_tbps: 1.008, bf16_tf: 330, fp8_tf: 661, fp4_tf: null },
    { id: 'rtx5090', name: 'GeForce RTX 5090', mem_gb: 32, mem_tbps: 1.792, bf16_tf: 419, fp8_tf: 838, fp4_tf: 1676 },
    { id: 'a100', name: 'A100 SXM 80GB', mem_gb: 80, mem_tbps: 2.04, bf16_tf: 312, fp8_tf: null, fp4_tf: null },
    { id: 'h100', name: 'H100 SXM', mem_gb: 80, mem_tbps: 3.35, bf16_tf: 989, fp8_tf: 1979, fp4_tf: null },
    { id: 'h200', name: 'H200 SXM', mem_gb: 141, mem_tbps: 4.8, bf16_tf: 989, fp8_tf: 1979, fp4_tf: null },
    { id: 'b200', name: 'B200 (HGX)', mem_gb: 180, mem_tbps: 8.0, bf16_tf: 2250, fp8_tf: 4500, fp4_tf: 9000 },
    { id: 'b300', name: 'B300 (Blackwell Ultra)', mem_gb: 288, mem_tbps: 8.0, bf16_tf: 2250, fp8_tf: 4500, fp4_tf: 15000 },
    { id: 'mi300x', name: 'Instinct MI300X', mem_gb: 192, mem_tbps: 5.3, bf16_tf: 1307, fp8_tf: 2615, fp4_tf: null },
    { id: 'mi355x', name: 'Instinct MI355X', mem_gb: 288, mem_tbps: 8.0, bf16_tf: 2500, fp8_tf: 5000, fp4_tf: 10000 },
    { id: 'ironwood', name: 'TPU v7 (Ironwood)', mem_gb: 192, mem_tbps: 7.37, bf16_tf: 2307, fp8_tf: 4614, fp4_tf: null }
  ];
  var LOCAL_DEVICES = [
    { id: 'm4max', name: 'Apple M4 Max (128 GB)', mem_gb: 128, mem_tbps: 0.546, bf16_tf: 34, fp8_tf: null, fp4_tf: null },
    { id: 'm3ultra', name: 'Apple M3 Ultra (512 GB)', mem_gb: 512, mem_tbps: 0.819, bf16_tf: 57, fp8_tf: null, fp4_tf: null }
  ];
  var KEEP = ['rtx4090', 'rtx5090', 'a100', 'h100', 'h200', 'b200', 'b300', 'mi300x', 'mi355x', 'ironwood'];
  var accelPromise = null;
  function accelerators() {
    if (accelPromise) return accelPromise;
    accelPromise = new Promise(function (resolve) {
      var done = function (list) { resolve(list.concat(LOCAL_DEVICES)); };
      try {
        var x = new XMLHttpRequest(); x.open('GET', BASE + '/assets/data/accelerators.json');
        x.onload = function () {
          try { var d = JSON.parse(x.responseText).chips.filter(function (c) { return KEEP.indexOf(c.id) !== -1 && c.mem_gb; }); done(d.length ? d : ACCEL_FALLBACK); }
          catch (e) { done(ACCEL_FALLBACK); }
        };
        x.onerror = function () { done(ACCEL_FALLBACK); }; x.send();
      } catch (e) { done(ACCEL_FALLBACK); }
    });
    return accelPromise;
  }
  var C = function (name) { return 'var(--' + name + ')'; };

  // =====================================================================
  //                                LABS
  // =====================================================================
  var LABS = {};

  // ---------------------------------------------------------- ngram / machine
  LABS.ngram = function (root) {
    var hero = root.dataset.variant === 'hero';
    var model = trainNgram(tokenizeWords(CORPUS)), rnd = mulberry32(Date.now() % 100000);
    var prompts = ['the model reads', 'attention lets every', 'the kv cache', 'a reasoning model', 'the router picks', 'the agent reads'];
    var tokens = tokenizeWords(prompts[0]), newIdx = -1, timer = null;
    var f, textBox, distBox, tempS, topkS, autoBtn, orderNote;
    if (hero) {
      root.classList.add('machine');
      root.innerHTML = '';
      root.appendChild(h('div', { class: 'mc-head' }, h('span', { class: 'dots' }, h('i'), h('i'), h('i')), h('span', { class: 'mc-t', text: 'next-token machine · a trigram LM' }), h('span', { class: 'mc-live', text: '● live' })));
      textBox = h('div', { class: 'mc-text', 'aria-live': 'polite' }); root.appendChild(textBox);
      var distWrap = h('div', { class: 'mc-dist' }); orderNote = h('span'); distWrap.appendChild(h('div', { class: 'mc-dist-label' }, h('span', { text: 'P(next token | context)' }), orderNote));
      distBox = h('div'); distWrap.appendChild(distBox); root.appendChild(distWrap);
      var ctl = h('div', { class: 'mc-controls' }); root.appendChild(ctl);
      f = { controls: ctl };
      root.appendChild(h('div', { class: 'mc-foot', text: 'A real (tiny) language model: it counts which word follows each pair of words in ~40 sentences from this guide. It sees only 2 tokens of context — highlighted in blue. A transformer sees all of them.' }));
    } else {
      f = frame(root, 'The next-token machine', 'A trigram model trained on ~40 sentences from this guide. It conditions on the last <b>two</b> tokens only (blue). Sample a few steps and watch it lose the thread.');
      textBox = h('div', { class: 'tokens-out', 'aria-live': 'polite' }); f.out.appendChild(textBox);
      orderNote = h('div', { class: 'lab-note' });
      distBox = h('div'); f.out.appendChild(distBox); f.out.appendChild(orderNote);
    }
    tempS = slider(f.controls, { label: 'temperature', min: 0, max: 2, step: 0.05, value: 0.8, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    topkS = slider(f.controls, { label: 'top-k (0 = off)', min: 0, max: 10, step: 1, value: 0, onchange: draw });
    var bar = barList(distBox);
    function current() { var nx = ngramNext(model, tokens); return { order: nx.order, dist: reshapeDist(nx.dist.slice(0, 40), tempS.get(), topkS.get()) }; }
    function draw() {
      textBox.innerHTML = '';
      var show = hero ? tokens.slice(-28) : tokens.slice(-22), off = tokens.length - show.length;
      show.forEach(function (t, i) {
        var idx = off + i, cls = 'tk' + (idx === newIdx ? ' new' : '') + (idx >= tokens.length - 2 ? ' ctx' : '');
        textBox.appendChild(h('span', { class: cls, text: (hero ? (/^[.,;:!?]$/.test(t) ? '' : ' ') : '') + t }));
      });
      if (hero) textBox.appendChild(h('span', { class: 'caret' }));
      var cur = current(), top = cur.dist.slice(0, hero ? 6 : 8);
      bar(top.map(function (d) { return { k: '"' + d.tok + '"', v: d.p.toFixed(3), w: d.p, cls: d.p === 0 ? 'off' : '' }; }));
      var on = ['', 'unigram (no context matched)', 'bigram (1-token context)', 'trigram (2-token context)'][cur.order];
      orderNote.textContent = hero ? on : 'Using the ' + on + '. Vocabulary: ' + Object.keys(model.uni).length + ' words.';
    }
    function step() {
      var cur = current(), probs = cur.dist.map(function (d) { return d.p; });
      var i = sampleIndex(probs, rnd); tokens.push(cur.dist[i].tok); newIdx = tokens.length - 1;
      if (tokens.length > 60) tokens = tokens.slice(-40), newIdx = tokens.length - 1;
      draw();
    }
    var row = h('div', { class: 'ctl-row' });
    row.appendChild(h('button', { class: 'lab-btn', type: 'button', text: 'step', onclick: step }));
    autoBtn = h('button', { class: 'lab-btn ghost', type: 'button', text: hero ? 'pause' : 'auto', onclick: function () { if (timer) { clearInterval(timer); timer = null; autoBtn.textContent = 'auto'; } else { timer = setInterval(step, 650); autoBtn.textContent = 'pause'; } } });
    row.appendChild(autoBtn);
    row.appendChild(h('button', { class: 'lab-btn ghost', type: 'button', text: 'new prompt', onclick: function () { tokens = tokenizeWords(prompts[Math.floor(rnd() * prompts.length)]); newIdx = -1; draw(); } }));
    f.controls.appendChild(row);
    if (hero) { for (var w = 0; w < 9; w++) step(); } else draw();
    if (hero && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      timer = setInterval(function () { step(); if (tokens.length > 34) { tokens = tokenizeWords(prompts[Math.floor(rnd() * prompts.length)]); newIdx = -1; } }, 1100);
    } else if (hero) { autoBtn.textContent = 'auto'; }
  };

  // ---------------------------------------------------------- rope
  LABS.rope = function (root) {
    var f = frame(root, 'RoPE rotations', 'A query at position <b>m</b> and a key at position <b>n</b> are rotated pair by pair. Move both by the same amount: the score is unchanged. Change the gap: it moves.');
    var d = 8, base = 10000;
    var mS = slider(f.controls, { label: 'query position m', min: 0, max: 64, value: 12, onchange: draw });
    var nS = slider(f.controls, { label: 'key position n', min: 0, max: 64, value: 9, onchange: draw });
    var bS = select(f.controls, { label: 'base θ', options: [['100', '100'], ['10000', '10,000 (original)'], ['500000', '500,000 (Llama 3)']], value: '10000', onchange: function () { base = parseFloat(bS.get()); draw(); } });
    var put = readouts(f.out, [{ id: 'score', k: 'q·k after RoPE', hot: true }, { id: 'gap', k: 'gap m − n' }, { id: 'raw', k: 'q·k without RoPE' }]);
    var ch = chart(f.out, 220);
    var q = [1, 0.3, 0.8, -0.2, 0.5, 0.6, 0.9, 0.1], k = [0.9, 0.2, 0.7, -0.1, 0.6, 0.5, 1.0, 0.2];
    function draw() {
      var m = mS.get(), n = nS.get(), s = ropeScore(q, k, m, n, base);
      put('score', s.toFixed(3), 'depends only on m − n'); put('gap', String(m - n), 'tokens'); put('raw', dot(q, k).toFixed(3), 'same for any m, n');
      var pts = []; for (var g = -40; g <= 40; g++) pts.push([g, ropeScore(q, k, g + 40, 40, base)]);
      var cols = ['s-a', 's-b', 's-c', 's-d'];
      var pairPts = [0, 1, 2, 3].map(function (i) { var arr = []; for (var g = 0; g <= 64; g++) arr.push([g, Math.cos(g * ropeFreq(i, d, base))]); return { pts: arr, cls: cols[i] + ' dash', label: 'pair ' + i }; });
      ch({ x: { min: -40, max: 40, label: 'gap m − n (tokens)', ticks: [-40, -20, 0, 20, 40], fmt: String }, y: { min: -3, max: 3.5, label: 'score', ticks: [-3, -1.5, 0, 1.5, 3], fmt: function (v) { return v.toFixed(1); } },
        series: [{ pts: pts, cls: 's-a', label: 'q·k vs gap' }], points: [{ x: m - n, y: s, label: (m - n) + ' → ' + s.toFixed(2) }] });
      void pairPts;
    }
    draw();
  };

  // ---------------------------------------------------------- params
  LABS.params = function (root) {
    var f = frame(root, 'Parameter counter', 'Pick a real model, then change any knob. Bars show where the parameters live; MoE presets also show <b>active</b> parameters per token.');
    var cfg = JSON.parse(JSON.stringify(MODELS['llama31-8b']));
    var pS = select(f.controls, { label: 'preset', options: Object.keys(MODELS).map(function (k) { return [k, MODELS[k].name]; }), value: 'llama31-8b', onchange: function () { cfg = JSON.parse(JSON.stringify(MODELS[pS.get()])); sync(); draw(); } });
    var dS = slider(f.controls, { label: 'd_model', min: 512, max: 16384, step: 128, value: cfg.d, onchange: rd });
    var LS = slider(f.controls, { label: 'layers', min: 2, max: 128, value: cfg.L, onchange: rd });
    var HS = slider(f.controls, { label: 'query heads', min: 1, max: 128, value: cfg.H, onchange: rd });
    var KS = slider(f.controls, { label: 'KV heads', min: 1, max: 128, value: cfg.Hkv, onchange: rd });
    var FS = slider(f.controls, { label: 'FFN width (dense)', min: 512, max: 65536, step: 256, value: cfg.dff, onchange: rd });
    var VOCABS = [[32000, 'Llama 2 / Mistral'], [50257, 'GPT-2'], [128256, 'Llama 3'], [129280, 'DeepSeek-V3'], [151936, 'Qwen3'], [163840, 'Kimi K2'], [201088, 'o200k (gpt-oss)'], [262144, 'Gemma 3 tokenizer']];
    var VS = select(f.controls, { label: 'vocabulary', options: VOCABS.map(function (v) { return [String(v[0]), v[0].toLocaleString('en-US') + ' · ' + v[1]]; }), value: String(cfg.V), onchange: rd });
    VS.set = function (v) { if (!VOCABS.some(function (x) { return x[0] === v; })) { VOCABS.push([v, 'preset']); VS.el.appendChild(h('option', { value: String(v), text: v.toLocaleString('en-US') })); } VS.el.value = String(v); };
    var tied = check(f.controls, { label: 'tie embedding & head', value: cfg.tied, onchange: rd });
    var put = readouts(f.out, [{ id: 'tot', k: 'total parameters', hot: true }, { id: 'act', k: 'active per token' }, { id: 'mem', k: 'BF16 weights' }]);
    var bars = barList(f.out); var note = h('div', { class: 'lab-note' }); f.out.appendChild(note);
    function sync() { dS.set(cfg.d); LS.set(cfg.L); HS.set(cfg.H); KS.set(cfg.Hkv); FS.set(cfg.dff); VS.set(cfg.V); }
    function rd() { cfg.d = dS.get(); cfg.L = LS.get(); cfg.H = HS.get(); cfg.Hkv = Math.min(KS.get(), cfg.H); cfg.dff = FS.get(); cfg.V = +VS.get(); cfg.tied = tied.get(); if (cfg.moe) cfg.moe.dffe = cfg.moe.dffe; draw(); }
    function draw() {
      var p = countParams(cfg), mx = p.total;
      put('tot', fmtNum(p.total), p.total.toLocaleString('en-US')); put('act', fmtNum(p.active), cfg.moe ? pct(p.active / p.total) + ' of total' : 'dense: all of them'); put('mem', fmtGB(p.total * 2), 'just to hold the weights');
      bars([
        { k: 'attention', v: pct(p.attn / mx), w: p.attn / mx, color: C('blue') },
        { k: cfg.moe ? 'experts (all)' : 'feed-forward', v: pct(p.ffn / mx), w: p.ffn / mx, color: C('verm') },
        { k: 'embedding', v: pct(p.emb / mx), w: p.emb / mx, color: C('teal') },
        { k: 'output head', v: pct(p.head / mx), w: p.head / mx, color: C('purple') },
        { k: 'norms', v: pct(p.norms / mx, 3), w: p.norms / mx, color: C('ink-3') }
      ]);
      var dh = cfg.dh || cfg.d / cfg.H;
      note.innerHTML = 'Per layer: attention <b>' + fmtNum(p.perLayerAttn) + '</b> (head dim ' + dh + (cfg.dh ? ', explicit' : '') + '), ' + (cfg.moe ? cfg.moe.E + ' experts × ' + fmtNum(3 * cfg.d * cfg.moe.dffe) + ', ' + cfg.moe.k + ' active' : 'FFN <b>' + fmtNum(3 * cfg.d * cfg.dff) + '</b> (SwiGLU: 3 × d × d_ff)') + '. KV cache: ' + fmtBytes(2 * cfg.L * cfg.Hkv * dh * 2) + ' per token in BF16.';
    }
    draw();
  };

  // ---------------------------------------------------------- attention
  LABS.attention = function (root) {
    var f = frame(root, 'Attention heatmap', 'Seven tokens with small hand-built query and key vectors (each key says what a token <i>is</i>; each query what it <i>looks for</i>). Rows are queries, columns keys; each row is a softmax.');
    var toks = ['The', 'cat', 'sat', 'because', 'it', 'was', 'tired'];
    //            det  noun anim verb conj pron aux adj
    var K = [[3, 0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 0, 0, 3]];
    var Q = [[1, 1, 0, 0, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1.2, 0, 0, 0, 0, 0], [0, 0, 0, 1.2, 0, 0, 0, 0], [0, 1, 1.4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.4, 0, 0], [0, 0.6, 1, 0, 0, 0.8, 0.4, 0]];
    var qi = 4;
    var tS = slider(f.controls, { label: 'temperature (× √d)', min: 0.2, max: 4, step: 0.05, value: 1, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var causal = check(f.controls, { label: 'causal mask', value: true, onchange: draw });
    var qSel = select(f.controls, { label: 'query token', options: toks.map(function (t, i) { return [String(i), i + ' · ' + t]; }), value: '4', onchange: function () { qi = +qSel.get(); draw(); } });
    var put = readouts(f.out, [{ id: 'top', k: 'strongest key for query', hot: true }, { id: 'ent', k: 'row entropy' }]);
    var heat = h('div', { class: 'heat' }); f.out.appendChild(heat);
    var note = h('div', { class: 'lab-note', html: 'Pick <b>"it"</b>: its query asks for an animate noun, so it attends to <b>"cat"</b> — the pronoun-resolution pattern real heads learn. Raise the temperature and every row flattens toward uniform.' });
    f.out.appendChild(note);
    function draw() {
      var d = 8, T = tS.get() * Math.sqrt(d), n = toks.length, rows = [];
      for (var i = 0; i < n; i++) {
        var sc = []; for (var j = 0; j < n; j++) sc.push(causal.get() && j > i ? -Infinity : dot(Q[i], K[j]));
        rows.push(softmax(sc, T));
      }
      heat.innerHTML = ''; heat.style.gridTemplateColumns = '4.2rem repeat(' + n + ', minmax(0, 1fr))';
      heat.appendChild(h('span'));
      toks.forEach(function (t, j) { heat.appendChild(h('span', { class: 'hh' + (j === rows[qi].indexOf(Math.max.apply(null, rows[qi])) ? ' sel' : ''), text: t })); });
      rows.forEach(function (r, i) {
        var lab = h('span', { class: 'hr' + (i === qi ? ' sel' : ''), text: toks[i] }); lab.style.cursor = 'pointer'; lab.addEventListener('click', function () { qi = i; qSel.el.value = String(i); draw(); });
        heat.appendChild(lab);
        r.forEach(function (p) {
          var c = h('span', { class: 'hc', text: p >= 0.995 ? '1' : p < 0.005 ? '' : p.toFixed(2).replace(/^0/, '') });
          c.style.background = 'color-mix(in srgb, var(--' + (i === qi ? 'verm' : 'blue') + ') ' + Math.round(p * 92) + '%, var(--paper-2))';
          if (p > 0.55) c.style.color = '#fff';
          heat.appendChild(c);
        });
      });
      var r = rows[qi], best = r.indexOf(Math.max.apply(null, r)), H = 0; r.forEach(function (p) { if (p > 0) H -= p * Math.log2(p); });
      put('top', '"' + toks[best] + '" ' + pct(r[best], 0), 'for query "' + toks[qi] + '"'); put('ent', H.toFixed(2) + ' bits', 'max ' + Math.log2(causal.get() ? qi + 1 : n).toFixed(2) + ' bits');
    }
    draw();
  };

  // ---------------------------------------------------------- kvcache
  LABS.kvcache = function (root) {
    var f = frame(root, 'KV-cache calculator', 'Bytes per token = layers × (2 × KV heads × head dim) × bytes — or latent + RoPE dims for MLA; windowed layers stop growing at the window.');
    var cfg = JSON.parse(JSON.stringify(KV_PRESETS['llama31-8b']));
    var pS = select(f.controls, { label: 'model', options: Object.keys(KV_PRESETS).map(function (k) { return [k, KV_PRESETS[k].name]; }), value: 'llama31-8b', onchange: function () { cfg = JSON.parse(JSON.stringify(KV_PRESETS[pS.get()])); draw(); } });
    var tS = slider(f.controls, { label: 'context (tokens)', min: 10, max: 20, step: 0.25, value: 15, map: function (v) { return Math.round(Math.pow(2, v)); }, fmt: function (v) { return v >= 1024 ? Math.round(v / 1024) + 'K' : String(v); }, onchange: draw });
    var bS = slider(f.controls, { label: 'concurrent sequences', min: 1, max: 256, value: 1, onchange: draw });
    var dt = select(f.controls, { label: 'cache dtype', options: [['2', 'BF16'], ['1', 'FP8'], ['0.5', '4-bit']], value: '2', onchange: draw });
    var put = readouts(f.out, [{ id: 'tok', k: 'per token', hot: true }, { id: 'seq', k: 'one sequence' }, { id: 'all', k: 'all sequences' }]);
    var bars = barList(f.out); var note = h('div', { class: 'lab-note' }); f.out.appendChild(note);
    function draw() {
      cfg.bytes = parseFloat(dt.get());
      var T = tS.get(), B = bS.get(), one = kvBytes(cfg, T);
      put('tok', fmtBytes(one / T), 'averaged over ' + T.toLocaleString('en-US') + ' tokens'); put('seq', fmtBytes(one), ''); put('all', fmtBytes(one * B), B > 1 ? B + ' × ' + fmtBytes(one) : '');
      var vs = ['mha', 'gqa', 'mqa', 'mla'].map(function (v) {
        var c = JSON.parse(JSON.stringify(cfg)); c.variant = v; c.window = 0; c.localLayers = 0;
        if (v === 'gqa' && cfg.variant !== 'gqa') c.Hkv = 8; if (v === 'mla') { c.dc = c.dc || 512; c.dr = c.dr || 64; }
        return { v: v, b: kvBytes(c, T) * B };
      });
      var mx = Math.max.apply(null, vs.map(function (x) { return x.b; }));
      bars(vs.map(function (x) { return { k: { mha: 'MHA (' + cfg.H + ' KV heads)', gqa: 'GQA (' + (cfg.variant === 'gqa' ? cfg.Hkv : 8) + ' KV heads)', mqa: 'MQA (1 KV head)', mla: 'MLA (512 + 64)' }[x.v], v: fmtBytes(x.b), w: x.b / mx, cls: x.v === cfg.variant ? 'picked' : '', color: x.v === cfg.variant ? C('verm') : C('blue') }; }));
      note.innerHTML = 'Same depth, same context, four attention designs. ' + (cfg.window ? '<b>' + cfg.localLayers + ' of ' + cfg.L + '</b> layers are windowed (' + cfg.window + ' tokens) in the selected model. ' : '') + 'An 80 GB GPU holds ' + fmtBytes(80e9) + '.';
    }
    draw();
  };

  // ---------------------------------------------------------- compute
  LABS.compute = function (root) {
    var f = frame(root, 'Training compute budget', 'C ≈ 6 · N<sub>active</sub> · D. Pick a model size, a token budget and a cluster; MFU is the fraction of dense peak you actually sustain.');
    accelerators().then(function (acc) {
      var nS = slider(f.controls, { label: 'active parameters', min: 8, max: 12, step: 0.05, value: 9.9, map: function (v) { return Math.pow(10, v); }, fmt: fmtNum, onchange: draw });
      var dS = slider(f.controls, { label: 'training tokens', min: 10, max: 13.7, step: 0.05, value: 13.18, map: function (v) { return Math.pow(10, v); }, fmt: fmtNum, onchange: draw });
      var gS = select(f.controls, { label: 'accelerator', options: acc.filter(function (a) { return a.bf16_tf > 100; }).map(function (a) { return [a.id, a.name]; }), value: 'h100', onchange: draw });
      var pr = select(f.controls, { label: 'precision', options: [['bf16', 'BF16'], ['fp8', 'FP8']], value: 'bf16', onchange: draw });
      var mS = slider(f.controls, { label: 'MFU', min: 0.15, max: 0.6, step: 0.01, value: 0.4, fmt: pct, onchange: draw });
      var cS = slider(f.controls, { label: 'GPUs', min: 3, max: 17, step: 1, value: 10, map: function (v) { return Math.pow(2, v); }, fmt: function (v) { return v.toLocaleString('en-US'); }, onchange: draw });
      var $S = slider(f.controls, { label: '$ per GPU-hour', min: 1, max: 6, step: 0.1, value: 2.5, fmt: function (v) { return '$' + v.toFixed(2); }, onchange: draw });
      var put = readouts(f.out, [{ id: 'f', k: 'training FLOPs', hot: true }, { id: 'h', k: 'GPU-hours' }, { id: 'd', k: 'wall-clock' }, { id: 'c', k: 'compute cost' }, { id: 'r', k: 'tokens / parameter' }]);
      var note = h('div', { class: 'lab-note' }); f.out.appendChild(note);
      function draw() {
        var a = acc.filter(function (x) { return x.id === gS.get(); })[0], peak = pr.get() === 'fp8' && a.fp8_tf ? a.fp8_tf : a.bf16_tf;
        var N = nS.get(), D = dS.get(), F = trainFlops(N, D), gh = gpuHours(F, peak, mS.get()), days = gh / cS.get() / 24;
        put('f', fmtSci(F), ''); put('h', fmtNum(gh), 'at ' + Math.round(peak * mS.get()) + ' TFLOP/s each');
        put('d', days >= 1 ? days.toFixed(days < 10 ? 1 : 0) + ' days' : (days * 24).toFixed(1) + ' hours', 'on ' + cS.get().toLocaleString('en-US') + ' GPUs, no failures');
        put('c', '$' + fmtNum(gh * $S.get()), 'GPU time only'); put('r', fmtNum(D / N), 'Chinchilla-optimal ≈ 20');
        note.innerHTML = 'Reference points: Llama 3.1 405B ≈ 3.8×10²⁵ FLOPs (30.8M H100-hours reported); DeepSeek-V3 ≈ 3.3×10²⁴ (2.79M H800-hours). Peak used: <b>' + peak + ' TFLOP/s</b> dense ' + (pr.get() === 'fp8' && a.fp8_tf ? 'FP8' : 'BF16') + '.';
      }
      draw();
    });
  };

  // ---------------------------------------------------------- vram (fine-tuning)
  LABS.vram = function (root) {
    var f = frame(root, 'Fine-tuning memory', 'Mixed-precision AdamW costs 16 bytes per trained parameter. Activations use the Korthikanti et al. estimate (34·s·b·h bytes per layer; 2·s·b·h with recomputation). The fp32 logits for the loss are shown separately.');
    var cfg = MODELS['llama31-8b'];
    var pS = select(f.controls, { label: 'model', options: [['llama31-8b', 'Llama 3.1 8B'], ['qwen3-8b', 'Qwen3-8B'], ['llama31-70b', 'Llama 3.1 70B'], ['qwen3-0.6b', 'Qwen3-0.6B']], value: 'llama31-8b', onchange: function () { cfg = MODELS[pS.get()]; draw(); } });
    var mS = select(f.controls, { label: 'method', options: [['full', 'Full fine-tuning'], ['lora', 'LoRA (BF16 base)'], ['qlora', 'QLoRA (NF4 base)']], value: 'qlora', onchange: draw });
    var rS = slider(f.controls, { label: 'LoRA rank (all-linear)', min: 1, max: 256, value: 16, onchange: draw });
    var sS = slider(f.controls, { label: 'sequence length', min: 9, max: 15, step: 1, value: 11, map: function (v) { return Math.pow(2, v); }, fmt: function (v) { return v.toLocaleString('en-US'); }, onchange: draw });
    var bS = slider(f.controls, { label: 'micro-batch', min: 1, max: 32, value: 1, onchange: draw });
    var ck = check(f.controls, { label: 'gradient checkpointing', value: true, onchange: draw });
    var fu = check(f.controls, { label: 'fused / chunked cross-entropy', value: false, onchange: draw });
    var put = readouts(f.out, [{ id: 't', k: 'estimated peak', hot: true }, { id: 'g', k: 'fits on' }]);
    var stack = h('div', { class: 'stack' }); f.out.appendChild(stack);
    var leg = h('div', { class: 'legend' }); f.out.appendChild(leg);
    var bars = barList(f.out);
    var parts = [['weights', 'blue'], ['grads', 'teal'], ['optim', 'purple'], ['acts', 'verm'], ['logits', 'yellow'], ['overhead', 'ink-4']];
    leg.innerHTML = parts.map(function (p) { return '<span><i style="background:var(--' + p[1] + ')"></i>' + p[0] + '</span>'; }).join('');
    function draw() {
      var p = countParams(cfg), loraP = rS.get() * loraPerRank(cfg, ['q', 'k', 'v', 'o', 'gate', 'up', 'down']);
      var r = finetuneMem({ N: p.total, embParams: p.emb + p.head, method: mS.get(), loraP: mS.get() === 'full' ? 0 : loraP, seq: sS.get(), batch: bS.get(), d: cfg.d, L: cfg.L, V: cfg.V, ckpt: ck.get(), fusedCE: fu.get(), overhead: 1.5e9 });
      stack.innerHTML = '';
      parts.forEach(function (q) { var i = h('i'); i.style.flexBasis = (r[q[0]] / r.total * 100).toFixed(2) + '%'; i.style.background = 'var(--' + q[1] + ')'; i.title = q[0] + ' ' + fmtGB(r[q[0]]); stack.appendChild(i); });
      put('t', fmtGB(r.total), mS.get() === 'full' ? 'shard across GPUs with FSDP' : fmtNum(loraP) + ' trainable');
      var fits = r.total < 16e9 ? '16 GB consumer GPU' : r.total < 24e9 ? '24 GB GPU' : r.total < 48e9 ? '48 GB GPU' : r.total < 80e9 ? '80 GB GPU' : Math.ceil(r.total / 72e9) + '× 80 GB (sharded)';
      put('g', fits, '');
      var mx = r.total;
      bars(parts.map(function (q) { return { k: q[0], v: fmtGB(r[q[0]]), w: r[q[0]] / mx, color: 'var(--' + q[1] + ')' }; }));
    }
    draw();
  };

  // ---------------------------------------------------------- lora
  LABS.lora = function (root) {
    var f = frame(root, 'The low-rank update', 'Left: trainable parameters for LoRA on a real model. Right: if the "true" fine-tuning update has a power-law spectrum, how much of its energy does rank r capture?');
    var cfg = MODELS['llama31-8b'];
    var rS = slider(f.controls, { label: 'rank r', min: 1, max: 256, value: 16, onchange: draw });
    var tS = select(f.controls, { label: 'target modules', options: [['qv', 'q_proj, v_proj'], ['attn', 'q, k, v, o'], ['all', 'all linear (incl. MLP)']], value: 'all', onchange: draw });
    var dS = slider(f.controls, { label: 'spectrum decay', min: 0.3, max: 2, step: 0.05, value: 0.8, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var put = readouts(f.out, [{ id: 'p', k: 'trainable (Llama 3.1 8B)', hot: true }, { id: 's', k: 'share of model' }, { id: 'e', k: 'update energy captured' }]);
    var ch = chart(f.out, 220);
    function draw() {
      var t = { qv: ['q', 'v'], attn: ['q', 'k', 'v', 'o'], all: ['q', 'k', 'v', 'o', 'gate', 'up', 'down'] }[tS.get()];
      var per = loraPerRank(cfg, t), r = rS.get(), P = per * r, tot = countParams(cfg).total;
      put('p', fmtNum(P), per.toLocaleString('en-US') + ' per unit rank'); put('s', pct(P / tot, 2), 'adapter ≈ ' + fmtBytes(P * 2) + ' in BF16');
      var n = 4096, e = energyCaptured(r, dS.get(), n); put('e', pct(e), 'of a 4096×4096 update');
      var pts = []; [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096].forEach(function (k) { pts.push([k, energyCaptured(k, dS.get(), n)]); });
      ch({ x: { min: 1, max: 4096, log: true, label: 'rank r (log)', ticks: [1, 4, 16, 64, 256, 1024, 4096], fmt: String }, y: { min: 0, max: 1, label: 'energy', ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (v) { return pct(v, 0); } }, series: [{ pts: pts, cls: 's-b' }], points: [{ x: r, y: e, label: 'r = ' + r }] });
    }
    draw();
  };

  // ---------------------------------------------------------- grpo
  LABS.grpo = function (root) {
    var f = frame(root, 'GRPO groups', 'Sample G answers per prompt, reward 1 if correct, normalise within the group. If every answer agrees, all advantages are zero — the prompt teaches nothing.');
    var gS = slider(f.controls, { label: 'group size G', min: 2, max: 32, value: 8, onchange: draw });
    var pS = slider(f.controls, { label: 'model success rate p', min: 0, max: 1, step: 0.01, value: 0.6, fmt: pct, onchange: draw });
    var seed = 3, rs = [];
    f.controls.appendChild(h('button', { class: 'lab-btn', type: 'button', text: 'resample group', onclick: function () { seed++; draw(); } }));
    var put = readouts(f.out, [{ id: 'z', k: 'prompts with zero gradient', hot: true }, { id: 'm', k: 'group mean reward' }]);
    var groupBox = h('div', { class: 'tokens-out' }); f.out.appendChild(groupBox);
    var ch = chart(f.out, 200);
    function draw() {
      var G = gS.get(), p = pS.get(), rnd = mulberry32(seed * 977 + G);
      rs = []; for (var i = 0; i < G; i++) rs.push(rnd() < p ? 1 : 0);
      var adv = groupAdvantages(rs);
      groupBox.innerHTML = '';
      rs.forEach(function (r, i) { var t = h('span', { class: 'tk' + (r ? ' new' : ''), text: (r ? '✓ ' : '✗ ') + (adv[i] >= 0 ? '+' : '') + adv[i].toFixed(2) }); groupBox.appendChild(t); });
      put('z', pct(zeroGradFraction(p, G)), 'p^G + (1 − p)^G'); put('m', (sum(rs) / G).toFixed(2), 'baseline for this group');
      var pts = []; for (var q = 0; q <= 1.0001; q += 0.02) pts.push([q, zeroGradFraction(q, G)]);
      ch({ x: { min: 0, max: 1, label: 'success rate p', ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (v) { return pct(v, 0); } }, y: { min: 0, max: 1, label: 'zero-grad', ticks: [0, 0.5, 1], fmt: function (v) { return pct(v, 0); } }, series: [{ pts: pts, cls: 's-a' }], points: [{ x: p, y: zeroGradFraction(p, G), label: 'G = ' + G }] });
    }
    draw();
  };

  // ---------------------------------------------------------- sampling
  LABS.sampling = function (root) {
    var f = frame(root, 'Sampling shaper', 'Next-token candidates after "The capital of France is". Temperature reshapes; top-k, top-p and min-p cut the tail. Struck-out rows can never be sampled.');
    var words = [' Paris', ' a', ' the', ' known', ' located', ' Lyon', ' famous', ' not', ' one', ' home', ' beautiful', ' Marseille'];
    var logits = [6.2, 3.1, 2.8, 2.4, 2.0, 1.7, 1.5, 1.1, 0.9, 0.6, 0.4, 0.1];
    var tS = slider(f.controls, { label: 'temperature', min: 0, max: 2, step: 0.05, value: 1, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var kS = slider(f.controls, { label: 'top-k (0 = off)', min: 0, max: 12, value: 0, onchange: draw });
    var pS = slider(f.controls, { label: 'top-p', min: 0.05, max: 1, step: 0.01, value: 1, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var mS = slider(f.controls, { label: 'min-p', min: 0, max: 0.5, step: 0.01, value: 0, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var rnd = mulberry32(42), last = -1;
    f.controls.appendChild(h('button', { class: 'lab-btn', type: 'button', text: 'sample', onclick: function () { last = sampleIndex(cur(), rnd); draw(); } }));
    var put = readouts(f.out, [{ id: 'n', k: 'tokens still possible', hot: true }, { id: 'h', k: 'entropy' }, { id: 's', k: 'last sample' }]);
    var bars = barList(f.out);
    function cur() { return shapeLogits(logits, { T: tS.get(), topK: kS.get(), topP: pS.get(), minP: mS.get() }); }
    function draw() {
      var p = cur(), n = p.filter(function (x) { return x > 0; }).length, H = 0; p.forEach(function (x) { if (x > 0) H -= x * Math.log2(x); });
      put('n', n + ' / ' + p.length, ''); put('h', H.toFixed(2) + ' bits', ''); put('s', last >= 0 ? '"' + words[last].trim() + '"' : '—', '');
      bars(words.map(function (w, i) { return { k: '"' + w.trim() + '"', v: p[i] > 0 ? p[i].toFixed(3) : 'cut', w: p[i], cls: (p[i] === 0 ? 'off' : '') + (i === last ? ' picked' : ''), color: i === last ? C('verm') : C('blue') }; }));
    }
    draw();
  };

  // ---------------------------------------------------------- specdec
  LABS.specdec = function (root) {
    var f = frame(root, 'Speculative decoding', 'Expected tokens per target pass = (1 − α<sup>K+1</sup>)/(1 − α). Speed-up divides that by the cost of drafting K tokens (c = draft step cost relative to a target step).');
    var aS = slider(f.controls, { label: 'acceptance rate α', min: 0.3, max: 0.98, step: 0.01, value: 0.7, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var kS = slider(f.controls, { label: 'draft length K', min: 1, max: 12, value: 5, onchange: draw });
    var cS = slider(f.controls, { label: 'draft cost c', min: 0, max: 0.5, step: 0.01, value: 0.05, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var put = readouts(f.out, [{ id: 'e', k: 'tokens per target pass', hot: true }, { id: 's', k: 'speed-up (bandwidth-bound)' }]);
    var ch = chart(f.out, 220);
    function draw() {
      var a = aS.get(), K = kS.get(), c = cS.get();
      put('e', specTokens(a, K).toFixed(2), 'including the target\'s own token'); put('s', specSpeedup(a, K, c).toFixed(2) + '×', 'shrinks as batch size grows');
      var s1 = [], s2 = []; for (var k = 1; k <= 12; k++) { s1.push([k, specSpeedup(a, k, c)]); s2.push([k, specTokens(a, k)]); }
      ch({ x: { min: 1, max: 12, label: 'draft length K', ticks: [1, 3, 5, 7, 9, 11], fmt: String }, y: (function () { var mx = Math.max(4, Math.ceil(specTokens(a, 12))), t = []; for (var v = 0; v <= mx; v++) t.push(v); return { min: 0, max: mx, label: 'tokens / speed-up', ticks: t, fmt: String }; })(),
        legend: true, series: [{ pts: s1, cls: 's-a', label: 'speed-up' }, { pts: s2, cls: 's-b dash', label: 'tokens per target pass' }], points: [{ x: K, y: specSpeedup(a, K, c), label: specSpeedup(a, K, c).toFixed(2) + '×' }] });
    }
    draw();
  };

  // ---------------------------------------------------------- paged
  LABS.paged = function (root) {
    var f = frame(root, 'Paged vs contiguous KV', 'The same stream of requests (log-normal lengths) admitted into a fixed KV budget. Contiguous allocation reserves the maximum length per request; paging allocates blocks on demand.');
    var mS = slider(f.controls, { label: 'max length reserved', min: 2048, max: 32768, step: 2048, value: 8192, fmt: function (v) { return v.toLocaleString('en-US'); }, onchange: draw });
    var lS = slider(f.controls, { label: 'mean request length', min: 128, max: 8192, step: 128, value: 1200, fmt: function (v) { return v.toLocaleString('en-US'); }, onchange: draw });
    var bS = select(f.controls, { label: 'block size', options: [['16', '16 tokens'], ['32', '32'], ['256', '256'], ['1024', '1024']], value: '16', onchange: draw });
    var put = readouts(f.out, [{ id: 'c', k: 'contiguous: admitted', hot: false }, { id: 'p', k: 'paged: admitted', hot: true }, { id: 'w', k: 'waste (contig → paged)' }]);
    var mapC = h('div', { class: 'mem-map' }), mapP = h('div', { class: 'mem-map' });
    f.out.appendChild(h('div', { class: 'lab-note', html: '<b>Contiguous</b> — every square is 1/256 of the KV budget; colour = holds real tokens, pale = reserved but empty' })); f.out.appendChild(mapC);
    f.out.appendChild(h('div', { class: 'lab-note', html: '<b>Paged</b>' })); f.out.appendChild(mapP);
    function paint(map, held, used, cap, color) {
      map.innerHTML = ''; var cells = 256, hc = Math.round(held / cap * cells), uc = Math.round(used / cap * cells);
      for (var i = 0; i < cells; i++) { var c = h('i'); if (i < uc) c.style.background = color; else if (i < hc) c.style.background = 'color-mix(in srgb, ' + color + ' 22%, var(--paper-3))'; map.appendChild(c); }
    }
    function draw() {
      var cap = 1 << 19, lens = requestStream(400, 5, lS.get(), mS.get()), r = allocate(lens, cap, +bS.get(), mS.get());
      put('c', r.contiguous.n + ' requests', 'tokens used ' + pct(r.contiguous.used / cap)); put('p', r.paged.n + ' requests', 'tokens used ' + pct(r.paged.used / cap)); put('w', pct(r.contiguous.waste, 0) + ' → ' + pct(r.paged.waste, 1), 'of allocated memory');
      paint(mapC, r.contiguous.held, r.contiguous.used, cap, C('blue')); paint(mapP, r.paged.held, r.paged.used, cap, C('verm'));
    }
    draw();
  };

  // ---------------------------------------------------------- bits
  LABS.bits = function (root) {
    var f = frame(root, 'Number-format explorer', 'Type a value. Each row shows the nearest representable number (round-to-nearest-even), its bits — <b style="color:var(--pink)">sign</b>, <b style="color:var(--blue)">exponent</b>, <b style="color:var(--green)">mantissa</b> — and the relative error. FP8/6/4 saturate at their max instead of overflowing (OCP convention); FP16 and BF16 overflow to ∞.', true);
    var inp = h('input', { type: 'text', value: '0.3', 'aria-label': 'value' });
    f.controls.appendChild(h('label', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', { text: 'value' })), inp));
    var presets = h('div', { class: 'ctl-row' });
    ['0.3', '3.14159', '100000', '1.001', '448', '0.0001', '-6.5'].forEach(function (v) { presets.appendChild(h('button', { class: 'lab-btn ghost', type: 'button', text: v, onclick: function () { inp.value = v; draw(); } })); });
    f.controls.appendChild(presets);
    var box = h('div'); f.out.appendChild(box);
    inp.addEventListener('input', draw);
    function bitsOf(fmt, code) {
      var n = 1 + fmt.e + fmt.m, s = [];
      for (var i = n - 1; i >= 0; i--) s.push(Math.floor(code / Math.pow(2, i)) % 2);
      return s;
    }
    function draw() {
      var x = parseFloat(inp.value); box.innerHTML = '';
      if (!isFinite(x)) { box.appendChild(h('div', { class: 'lab-note', text: 'Enter a finite number.' })); return; }
      Object.keys(FORMATS).forEach(function (k) {
        var fmt = FORMATS[k], r = fpEncode(fmt, x), b = bitsOf(fmt, r.code), strip = h('div', { class: 'bitstrip' });
        b.forEach(function (bit, i) { strip.appendChild(h('b', { class: i === 0 ? 's' : i <= fmt.e ? 'e' : 'm', text: String(bit) })); });
        var err = x === 0 ? 0 : Math.abs(r.value - x) / Math.abs(x);
        var shown = isFinite(r.value) ? (Math.abs(r.value) >= 1e6 || (Math.abs(r.value) < 1e-4 && r.value !== 0) ? r.value.toExponential(6) : String(+r.value.toPrecision(10))) : String(r.value);
        box.appendChild(h('div', { class: 'fmt-row' }, h('span', { class: 'fk', text: fmt.name }), h('div', {}, strip, h('span', { class: 'fv', text: '= ' + shown + (isFinite(r.value) ? '   (rel. error ' + (err === 0 ? '0' : err < 1e-4 ? err.toExponential(1) : pct(err, 2)) + ')' : '   overflow') }))));
      });
    }
    draw();
  };

  // ---------------------------------------------------------- quant
  LABS.quant = function (root) {
    var f = frame(root, 'Quantization bench', 'A row of 4,096 weight-like values (σ = 0.02) with injected outliers. Round-to-nearest at the chosen bit-width and granularity; the error is relative RMS.');
    var bS = select(f.controls, { label: 'bits', options: [['8', 'INT8'], ['4', 'INT4'], ['3', 'INT3'], ['2', 'INT2']], value: '4', onchange: draw });
    var mS = select(f.controls, { label: 'mode', options: [['sym', 'symmetric'], ['asym', 'asymmetric (zero point)']], value: 'sym', onchange: draw });
    var gS = select(f.controls, { label: 'granularity', options: [['0', 'per-tensor (one scale)'], ['1024', 'groups of 1024'], ['128', 'groups of 128'], ['32', 'groups of 32 (MX-like)'], ['16', 'groups of 16 (NVFP4-like)']], value: '128', onchange: draw });
    var oS = slider(f.controls, { label: 'outlier fraction', min: 0, max: 0.02, step: 0.0005, value: 0.002, fmt: function (v) { return pct(v, 2); }, onchange: draw });
    var xS = slider(f.controls, { label: 'outlier magnitude', min: 1, max: 100, value: 30, fmt: function (v) { return v + '×'; }, onchange: draw });
    var put = readouts(f.out, [{ id: 'e', k: 'relative error', hot: true }, { id: 'b', k: 'bits per weight (incl. FP16 scales)' }]);
    var bars = barList(f.out); var note = h('div', { class: 'lab-note' }); f.out.appendChild(note);
    function draw() {
      var w = weightRow(4096, 9, oS.get(), xS.get()), bits = +bS.get(), g = +gS.get(), mode = mS.get();
      var e = relError(w, quantize(w, bits, mode, g)), overhead = g > 0 ? (mode === 'asym' ? 32 : 16) / g : 0;
      put('e', pct(e, 2), ''); put('b', (bits + overhead).toFixed(2), g > 0 ? 'scale' + (mode === 'asym' ? ' + zero' : '') + ' every ' + g : 'one scale for all');
      var rows = [['0', 'per-tensor'], ['1024', 'g = 1024'], ['128', 'g = 128'], ['32', 'g = 32'], ['16', 'g = 16']].map(function (q) { return { g: +q[0], k: q[1], e: relError(w, quantize(w, bits, mode, +q[0])) }; });
      var mx = Math.max.apply(null, rows.map(function (r) { return r.e; }));
      bars(rows.map(function (r) { return { k: r.k, v: pct(r.e, 1), w: r.e / mx, cls: r.g === g ? 'picked' : '', color: r.g === g ? C('verm') : C('blue') }; }));
      note.innerHTML = 'Smaller groups confine each outlier to fewer neighbours. Below 4 bits even small groups struggle — that is where GPTQ, AWQ, rotations and codebooks (Chapter 21) earn their keep.';
    }
    draw();
  };

  // ---------------------------------------------------------- fit
  var FIT_MODELS = [
    { id: 'llama8', name: 'Llama 3.1 8B (dense)', N: 8.03e9, act: 8.03e9, kv: 'llama31-8b' },
    { id: 'qwen30', name: 'Qwen3-30B-A3B (MoE)', N: 30.5e9, act: 3.3e9, kv: 'qwen3-30b-a3b' },
    { id: 'gemma27', name: 'Gemma 3 27B (dense)', N: 27.4e9, act: 27.4e9, kv: 'gemma3-27b' },
    { id: 'llama70', name: 'Llama 3.1 70B (dense)', N: 70.6e9, act: 70.6e9, kv: 'llama31-70b' },
    { id: 'gptoss', name: 'gpt-oss-120b (MoE)', N: 117e9, act: 5.1e9, kv: 'gpt-oss-120b' },
    { id: 'qwen235', name: 'Qwen3-235B-A22B (MoE)', N: 235e9, act: 22e9, kv: 'qwen3-235b' },
    { id: 'dsv3', name: 'DeepSeek-V3 (MoE, MLA)', N: 671e9, act: 37e9, kv: 'deepseek-v3' }
  ];
  LABS.fit = function (root) {
    var f = frame(root, 'Will it fit? How fast?', 'Weights at the chosen precision, plus a BF16 KV cache, against the device memory. Single-stream decode reads the active weights and the cache once per token, so speed ≲ bandwidth ÷ bytes.');
    accelerators().then(function (acc) {
      var mS = select(f.controls, { label: 'model', options: FIT_MODELS.map(function (m) { return [m.id, m.name]; }), value: 'llama70', onchange: draw });
      var pS = select(f.controls, { label: 'weights', options: [['16', 'BF16'], ['8.5', 'FP8 / Q8_0'], ['5.7', 'Q5_K_M (~5.7 b)'], ['4.8', 'Q4_K_M (~4.8 b)'], ['4.25', 'INT4 g128 / MXFP4'], ['3', '~3-bit']], value: '4.25', onchange: draw });
      var gS = select(f.controls, { label: 'device', options: acc.map(function (a) { return [a.id, a.name + ' · ' + a.mem_gb + ' GB']; }), value: 'h100', onchange: draw });
      var nS = slider(f.controls, { label: 'devices', min: 1, max: 8, value: 1, onchange: draw });
      var cS = slider(f.controls, { label: 'context (tokens)', min: 10, max: 18, step: 0.5, value: 13, map: function (v) { return Math.round(Math.pow(2, v)); }, fmt: function (v) { return v >= 1024 ? Math.round(v / 1024) + 'K' : String(v); }, onchange: draw });
      var put = readouts(f.out, [{ id: 'm', k: 'memory needed', hot: true }, { id: 'f', k: 'verdict' }, { id: 's', k: 'decode ceiling (1 stream)' }]);
      var stack = h('div', { class: 'stack' }); f.out.appendChild(stack);
      var note = h('div', { class: 'lab-note' }); f.out.appendChild(note);
      function draw() {
        var m = FIT_MODELS.filter(function (x) { return x.id === mS.get(); })[0], a = acc.filter(function (x) { return x.id === gS.get(); })[0], n = nS.get();
        var bpw = +pS.get(), wBytes = m.N * bpw / 8, kv = kvBytes(KV_PRESETS[m.kv], cS.get()), cap = a.mem_gb * 1e9 * n * 0.92, need = wBytes + kv;
        var read = m.act * bpw / 8 + kv, ceil = decodeCeiling(read, a.mem_tbps * n);
        put('m', fmtGB(need), fmtGB(wBytes) + ' weights + ' + fmtGB(kv) + ' cache');
        put('f', need <= cap ? 'fits ✓' : 'does not fit ✗', 'usable ≈ ' + fmtGB(cap) + ' (92%)');
        put('s', need <= cap ? Math.round(ceil) + ' tok/s' : '—', 'reads ' + fmtGB(read) + ' per token');
        stack.innerHTML = '';
        [[wBytes, 'blue', 'weights'], [kv, 'verm', 'KV cache'], [Math.max(0, cap - need), 'paper-3', 'free']].forEach(function (q) { var i = h('i'); i.style.flexBasis = (q[0] / Math.max(cap, need) * 100).toFixed(2) + '%'; i.style.background = 'var(--' + q[1] + ')'; i.title = q[2] + ' ' + fmtGB(q[0]); stack.appendChild(i); });
        note.innerHTML = (m.act < m.N ? 'MoE: only the <b>' + fmtNum(m.act) + '</b> active parameters are read per token at batch 1, but all ' + fmtNum(m.N) + ' must fit. ' : '') + 'Real runtimes typically reach 50–80% of the ceiling; ' + n + '-device figures assume perfect tensor parallelism.';
      }
      draw();
    });
  };

  // ---------------------------------------------------------- roofline
  LABS.roofline = function (root) {
    var f = frame(root, 'Roofline', 'Attainable FLOP/s = min(peak, intensity × bandwidth). LLM operations land on either side of the ridge depending on how many tokens share each weight read.');
    accelerators().then(function (acc) {
      var gS = select(f.controls, { label: 'accelerator', options: acc.filter(function (a) { return a.bf16_tf > 100; }).map(function (a) { return [a.id, a.name]; }), value: 'h100', onchange: draw });
      var pS = select(f.controls, { label: 'precision', options: [['bf16', 'BF16 (2 B)'], ['fp8', 'FP8 (1 B)'], ['fp4', 'FP4 (0.5 B)']], value: 'bf16', onchange: draw });
      var bS = slider(f.controls, { label: 'decode batch', min: 0, max: 10, step: 1, value: 5, map: function (v) { return Math.pow(2, v); }, onchange: draw });
      var put = readouts(f.out, [{ id: 'r', k: 'ridge point', hot: true }, { id: 'b', k: 'batched decode' }]);
      var ch = chart(f.out, 280);
      function draw() {
        var a = acc.filter(function (x) { return x.id === gS.get(); })[0], pr = pS.get();
        var peak = pr === 'fp4' ? (a.fp4_tf || a.fp8_tf || a.bf16_tf) : pr === 'fp8' ? (a.fp8_tf || a.bf16_tf) : a.bf16_tf, bytes = pr === 'fp4' ? 0.5 : pr === 'fp8' ? 1 : 2, bw = a.mem_tbps;
        var ridge = peak / bw, B = bS.get();
        var ops = [
          { k: 'decode, batch 1', I: gemmIntensity(1, 4096, 4096, bytes) },
          { k: 'decode, batch ' + B, I: gemmIntensity(B, 4096, 4096, bytes) },
          { k: 'attention decode (GQA-4)', I: 2 * 4 / bytes },
          { k: 'prefill, 4096 tokens', I: gemmIntensity(4096, 4096, 4096, bytes) }
        ];
        put('r', Math.round(ridge) + ' FLOP/B', peak + ' TFLOP/s ÷ ' + bw + ' TB/s');
        put('b', Math.round(attainable(ops[1].I, peak, bw)) + ' TFLOP/s', 'intensity ' + ops[1].I.toFixed(1) + ' → ' + pct(attainable(ops[1].I, peak, bw) / peak, 0) + ' of peak');
        var line = []; [0.1, 1, 10, ridge, 100, 1000, 10000].sort(function (x, y) { return x - y; }).forEach(function (i) { line.push([i, attainable(i, peak, bw)]); });
        ch({ x: { min: 0.1, max: 10000, log: true, label: 'arithmetic intensity (FLOP / byte, log)', ticks: [0.1, 1, 10, 100, 1000, 10000] }, y: { min: 0.1, max: Math.pow(10, Math.ceil(Math.log10(peak)) + 0.1), log: true, label: 'TFLOP/s (log)' },
          series: [{ pts: line, cls: 's-a' }], vlines: [{ x: ridge, label: 'ridge' }],
          points: ops.map(function (o, i) { var pos = [[10, 14, 'start'], [10, 14, 'start'], [-10, -8, 'end'], [-10, 36, 'end']][i]; return { x: o.I, y: attainable(o.I, peak, bw), label: o.k, fill: ['var(--blue)', 'var(--teal)', 'var(--purple)', 'var(--verm)'][i], dx: pos[0], dy: pos[1], anchor: pos[2] }; }) });
      }
      draw();
    });
  };

  // ---------------------------------------------------------- parallel
  LABS.parallel = function (root) {
    var f = frame(root, 'Parallelism memory planner', 'Per-GPU model state (16 bytes per parameter in mixed-precision AdamW) under tensor (TP), pipeline (PP) and data (DP) parallelism with ZeRO stages. Activations not included.');
    var nS = slider(f.controls, { label: 'parameters', min: 9, max: 12.3, step: 0.05, value: 10.85, map: function (v) { return Math.pow(10, v); }, fmt: fmtNum, onchange: draw });
    var tS = select(f.controls, { label: 'tensor parallel', options: [['1', '1'], ['2', '2'], ['4', '4'], ['8', '8']], value: '8', onchange: draw });
    var pS = select(f.controls, { label: 'pipeline parallel', options: [['1', '1'], ['2', '2'], ['4', '4'], ['8', '8'], ['16', '16']], value: '1', onchange: draw });
    var dS = select(f.controls, { label: 'data parallel', options: [['1', '1'], ['2', '2'], ['4', '4'], ['8', '8'], ['16', '16'], ['64', '64'], ['128', '128']], value: '8', onchange: draw });
    var zS = select(f.controls, { label: 'ZeRO stage', options: [['0', '0 (replicate)'], ['1', '1 (optimizer)'], ['2', '2 (+ gradients)'], ['3', '3 / FSDP (+ weights)']], value: '1', onchange: draw });
    var put = readouts(f.out, [{ id: 't', k: 'state per GPU', hot: true }, { id: 'g', k: 'GPUs' }, { id: 'f', k: 'on 80 GB' }]);
    var bars = barList(f.out);
    function draw() {
      var o = { N: nS.get(), tp: +tS.get(), pp: +pS.get(), dp: +dS.get(), zero: +zS.get() }, r = parallelMem(o);
      put('t', fmtGB(r.total), ''); put('g', String(o.tp * o.pp * o.dp), 'TP × PP × DP'); put('f', r.total < 60e9 ? 'room for activations' : r.total < 80e9 ? 'tight' : 'does not fit', '');
      var mx = Math.max(r.total, 80e9);
      bars([{ k: 'weights (BF16)', v: fmtGB(r.weights), w: r.weights / mx, color: C('blue') }, { k: 'gradients', v: fmtGB(r.grads), w: r.grads / mx, color: C('teal') }, { k: 'optimizer (FP32)', v: fmtGB(r.optim), w: r.optim / mx, color: C('purple') }, { k: '80 GB HBM', v: '80 GB', w: 80e9 / mx, color: C('ink-4') }]);
    }
    draw();
  };

  // ---------------------------------------------------------- servecost
  LABS.servecost = function (root) {
    var f = frame(root, 'Cost per million tokens', 'Each decode step reads the weights once (shared by the batch) and every sequence\'s KV cache; it takes the longer of that memory time and the compute time. Bigger batches amortise the weights — until compute or latency bites.');
    accelerators().then(function (acc) {
      var gS = select(f.controls, { label: 'accelerator', options: acc.filter(function (a) { return a.bf16_tf > 100; }).map(function (a) { return [a.id, a.name]; }), value: 'h100', onchange: draw });
      var mS = select(f.controls, { label: 'model (FP8 weights)', options: [['8', 'Llama 3.1 8B'], ['70', 'Llama 3.1 70B (×2 GPUs)'], ['moe', 'Qwen3-30B-A3B (MoE)']], value: '8', onchange: draw });
      var bS = slider(f.controls, { label: 'batch (concurrent streams)', min: 0, max: 9, step: 1, value: 5, map: function (v) { return Math.pow(2, v); }, onchange: draw });
      var cS = slider(f.controls, { label: 'context per stream', min: 10, max: 15, step: 1, value: 12, map: function (v) { return Math.pow(2, v); }, fmt: function (v) { return Math.round(v / 1024) + 'K'; }, onchange: draw });
      var $S = slider(f.controls, { label: '$ per GPU-hour', min: 1, max: 6, step: 0.1, value: 2.5, fmt: function (v) { return '$' + v.toFixed(2); }, onchange: draw });
      var put = readouts(f.out, [{ id: 'c', k: '$ per million output tokens', hot: true }, { id: 'p', k: 'per-stream speed' }, { id: 't', k: 'GPU throughput' }]);
      var ch = chart(f.out, 190), ch2 = chart(f.out, 170);
      function model() {
        var m = mS.get();
        // KV cache in FP8 (1 byte), like the weights
        if (m === '70') return { gpus: 2, N: 70.6e9, act: 70.6e9, kv: kvUnit(KV_PRESETS['llama31-70b']) * 80 / 2 };
        if (m === 'moe') return { gpus: 1, N: 30.5e9, act: 3.3e9, kv: kvUnit(KV_PRESETS['qwen3-30b-a3b']) * 48 / 2 };
        return { gpus: 1, N: 8.03e9, act: 8.03e9, kv: kvUnit(KV_PRESETS['llama31-8b']) * 32 / 2 };
      }
      function draw() {
        var a = acc.filter(function (x) { return x.id === gS.get(); })[0], m = model(), ctx = cS.get();
        var o = { weightBytes: m.N * 1.0 / m.gpus, kvBytesPerSeq: m.kv * ctx / m.gpus, bw: a.mem_tbps, peak: (a.fp8_tf || a.bf16_tf), activeN: m.act / m.gpus, effBw: 0.7, effCmp: 0.5 };
        var B = bS.get(), pt = servePoint(o, B), perGpu = pt.total, cost = costPerMillion($S.get() * m.gpus, perGpu * m.gpus);
        put('c', '$' + cost.toFixed(cost < 1 ? 3 : 2), pt.bound + '-bound at batch ' + B);
        put('p', Math.round(pt.perStream) + ' tok/s', 'what each user sees');
        put('t', fmtNum(perGpu * m.gpus) + ' tok/s', 'all streams, ' + m.gpus + ' GPU' + (m.gpus > 1 ? 's' : ''));
        var c1 = [], c2 = [];
        for (var e = 0; e <= 9; e++) { var b = Math.pow(2, e), p = servePoint(o, b); c1.push([b, costPerMillion($S.get() * m.gpus, p.total * m.gpus)]); c2.push([b, p.perStream]); }
        var xa = { min: 1, max: 512, log: true, label: 'batch — concurrent streams (log)', ticks: [1, 4, 16, 64, 256], fmt: String };
        ch({ x: xa, y: { min: 0.01, max: 100, log: true, label: '$ / M tokens', ticks: [0.01, 0.1, 1, 10, 100], fmt: function (v) { return '$' + fmtTick(v); } },
          series: [{ pts: c1, cls: 's-a' }], points: [{ x: B, y: cost, label: '$' + cost.toFixed(cost < 1 ? 3 : 2) }] });
        var hi = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, c2.map(function (q) { return q[1]; })) * 1.05)));
        ch2({ x: xa, y: { min: hi / 1000, max: hi, log: true, label: 'tok/s per stream' },
          series: [{ pts: c2, cls: 's-b' }], points: [{ x: B, y: pt.perStream, label: Math.round(pt.perStream) + ' tok/s', fill: 'var(--blue)' }] });
      }
      draw();
    });
  };

  // ---------------------------------------------------------- scaling
  LABS.scaling = function (root) {
    var f = frame(root, 'Chinchilla & over-training', 'The Chinchilla loss surface (Epoch AI\'s 2024 refit). At fixed compute, slide the model size off the optimum: loss barely moves near the optimum, while serving cost scales with parameters.');
    var cS = slider(f.controls, { label: 'training compute (FLOPs)', min: 20, max: 26, step: 0.1, value: 24, map: function (v) { return Math.pow(10, v); }, fmt: fmtSci, onchange: draw });
    var nS = slider(f.controls, { label: 'model size ÷ optimum', min: -1.5, max: 0.7, step: 0.05, value: 0, map: function (v) { return Math.pow(10, v); }, fmt: function (v) { return v.toFixed(2) + '×'; }, onchange: draw });
    var iS = slider(f.controls, { label: 'lifetime inference tokens', min: 9, max: 15, step: 0.1, value: 13, map: function (v) { return Math.pow(10, v); }, fmt: fmtNum, onchange: draw });
    var put = readouts(f.out, [{ id: 'o', k: 'compute-optimal', hot: true }, { id: 'n', k: 'your model' }, { id: 'l', k: 'loss penalty' }, { id: 't', k: 'lifetime FLOPs (train + serve)' }]);
    var ch = chart(f.out, 240);
    function draw() {
      var C0 = cS.get(), opt = chinOptimal(C0), N = opt.N * nS.get(), D = C0 / (6 * N), Li = iS.get();
      var Lopt = chinLoss(opt.N, opt.D), L = chinLoss(N, D);
      put('o', fmtNum(opt.N) + ' params', fmtNum(opt.D) + ' tokens · ' + (opt.D / opt.N).toFixed(0) + ' tok/param · loss ' + Lopt.toFixed(3));
      put('n', fmtNum(N) + ' params', fmtNum(D) + ' tokens · ' + fmtNum(D / N) + ' tok/param');
      put('l', '+' + (L - Lopt).toFixed(3) + ' nats', 'on Chinchilla\'s data');
      var life = C0 + 2 * N * Li, lifeOpt = C0 + 2 * opt.N * Li;
      put('t', fmtSci(life), (life < lifeOpt ? pct(1 - life / lifeOpt, 0) + ' less than the optimum model' : pct(life / lifeOpt - 1, 0) + ' more than the optimum model'));
      var pts = []; for (var e = -1.5; e <= 0.7001; e += 0.05) { var n = opt.N * Math.pow(10, e); pts.push([n, chinLoss(n, C0 / (6 * n))]); }
      var lo = Math.min.apply(null, pts.map(function (p) { return p[1]; })), hi = Math.max.apply(null, pts.map(function (p) { return p[1]; }));
      ch({ x: { min: opt.N * Math.pow(10, -1.5), max: opt.N * Math.pow(10, 0.7), log: true, label: 'parameters (log) at fixed compute — an isoFLOP curve', fmt: fmtNum }, y: { min: lo - 0.01, max: hi + 0.01, label: 'loss', fmt: function (v) { return v.toFixed(2); } },
        series: [{ pts: pts, cls: 's-b' }], points: [{ x: opt.N, y: Lopt, label: 'optimum', fill: 'var(--blue)', dy: -10 }, { x: N, y: L, label: 'yours', dy: 16 }] });
    }
    draw();
  };

  // ---------------------------------------------------------- moe
  LABS.moe = function (root) {
    var f = frame(root, 'Expert router', 'Tokens are routed to their top-k experts; experts that win tokens get better at them (rich get richer). Compare no balancing, an auxiliary loss, and DeepSeek-style bias balancing. Capacity limits drop overflow tokens.');
    var eS = select(f.controls, { label: 'experts', options: [['8', '8'], ['16', '16'], ['32', '32'], ['64', '64']], value: '16', onchange: draw });
    var kS = select(f.controls, { label: 'top-k', options: [['1', '1'], ['2', '2'], ['4', '4'], ['8', '8']], value: '2', onchange: draw });
    var mS = select(f.controls, { label: 'balancing', options: [['none', 'none'], ['aux', 'auxiliary loss'], ['bias', 'bias (aux-loss-free)']], value: 'none', onchange: draw });
    var cS = select(f.controls, { label: 'capacity factor', options: [['0', 'dropless'], ['1.25', '1.25'], ['2', '2.0']], value: '0', onchange: draw });
    var put = readouts(f.out, [{ id: 'i', k: 'max load ÷ mean', hot: true }, { id: 'd', k: 'dead experts' }, { id: 'x', k: 'tokens dropped' }]);
    var bars = barList(f.out); var ch = chart(f.out, 170);
    f.out.appendChild(h('div', { class: 'lab-note', html: 'With 256 tokens per step even a perfectly balanced router shows max/mean ≈ 1.2–2 from sampling noise alone. Collapse is the curve that keeps climbing; <b>dead experts</b> receive nothing and stop learning.' }));
    function draw() {
      var o = { E: +eS.get(), k: +kS.get(), tokens: 256, steps: 60, method: mS.get(), capacity: +cS.get(), eta: 0.04, alpha: 0.3, gamma: 0.15, seed: 7 };
      var r = simulateRouting(o), mean = o.tokens * o.k / o.E, mx = Math.max.apply(null, r.load);
      put('i', (mx / mean).toFixed(2) + '×', 'after ' + o.steps + ' steps'); put('d', String(r.dead), 'of ' + o.E); put('x', pct(r.dropped, 1), o.capacity ? 'capacity ' + o.capacity + '×' : 'dropless');
      bars(r.load.map(function (l, i) { return { k: 'expert ' + i, v: String(l), w: l / Math.max(mx, mean * 1.5), color: l > mean * 1.5 ? C('verm') : C('blue') }; }).slice(0, 16));
      var top = Math.max(4, Math.ceil(Math.max.apply(null, r.imbalance))), yt = []; for (var t = 1; t <= top; t += Math.max(1, Math.ceil((top - 1) / 5))) yt.push(t);
      ch({ x: { min: 1, max: o.steps, label: 'training step', ticks: [1, 20, 40, 60], fmt: String }, y: { min: 1, max: top, label: 'max/mean', ticks: yt, fmt: String }, series: [{ pts: r.imbalance.map(function (v, i) { return [i + 1, v]; }), cls: 's-a' }] });
    }
    draw();
  };

  // ---------------------------------------------------------- ssm
  LABS.ssm = function (root) {
    var f = frame(root, 'Fixed state vs KV cache', 'Store n random key→value facts, then query every key. A linear-attention state is a d×d matrix no matter how many facts; the KV cache keeps every key and value.');
    var dS = select(f.controls, { label: 'state width d', options: [['16', '16'], ['32', '32'], ['64', '64']], value: '64', onchange: draw });
    var nS = slider(f.controls, { label: 'facts stored n', min: 4, max: 256, step: 4, value: 64, onchange: draw });
    var put = readouts(f.out, [{ id: 'l', k: 'linear attention error' }, { id: 'd', k: 'delta rule error', hot: true }, { id: 's', k: 'softmax + KV cache error' }, { id: 'm', k: 'memory: state vs cache' }]);
    var ch = chart(f.out, 220);
    var cache = {};
    function err(n, d, rule) { var key = n + ':' + d + ':' + rule; if (!(key in cache)) cache[key] = recallError(n, d, rule, 1); return cache[key]; }
    function draw() {
      var d = +dS.get(), n = nS.get();
      put('l', pct(err(n, d, 'linear'), 0), 'relative error'); put('d', pct(err(n, d, 'delta'), 0), 'relative error'); put('s', pct(err(n, d, 'softmax'), 1), 'grows with every token');
      put('m', (d * d) + ' vs ' + (2 * n * d), 'numbers held');
      var xs = [4, 8, 16, 32, 64, 128, 256];
      ch({ x: { min: 4, max: 256, log: true, label: 'facts stored (log)', ticks: [4, 8, 16, 32, 64, 128, 256], fmt: String }, y: { min: 0, max: 3, label: 'recall error', ticks: [0, 0.5, 1, 1.5, 2, 2.5, 3], fmt: function (v) { return pct(v, 0); } },
        legend: true, series: [{ pts: xs.map(function (x) { return [x, err(x, d, 'linear')]; }), cls: 's-b', label: 'linear attention' }, { pts: xs.map(function (x) { return [x, err(x, d, 'delta')]; }), cls: 's-a', label: 'delta rule' }, { pts: xs.map(function (x) { return [x, err(x, d, 'softmax')]; }), cls: 's-c', label: 'softmax over KV cache' }],
        vlines: [{ x: d, label: 'n = d' }], points: [{ x: n, y: err(n, d, 'delta') }] });
    }
    draw();
  };

  // ---------------------------------------------------------- tts
  LABS.tts = function (root) {
    var f = frame(root, 'Test-time compute: voting', 'Each sample is right with probability p; wrong answers are spread over 5 distractors (concentration = how often the same wrong answer repeats). Majority vote, best-of-N with a noisy verifier, and pass@N (an oracle).');
    var pS = slider(f.controls, { label: 'per-sample accuracy p', min: 0.05, max: 0.95, step: 0.01, value: 0.4, fmt: pct, onchange: draw });
    var cS = slider(f.controls, { label: 'wrong-answer concentration', min: 0, max: 4, step: 0.1, value: 1, fmt: function (v) { return v.toFixed(1); }, onchange: draw });
    var vS = slider(f.controls, { label: 'verifier noise σ', min: 0.05, max: 2, step: 0.05, value: 0.6, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var put = readouts(f.out, [{ id: 'm', k: 'majority @ 32', hot: true }, { id: 'b', k: 'best-of-32' }, { id: 'p', k: 'pass@32 (oracle)' }]);
    var ch = chart(f.out, 240);
    function draw() {
      var r = votingCurve({ p: pS.get(), concentration: cS.get(), noise: vS.get(), maxN: 32, trials: 1500, seed: 11 });
      put('m', pct(r.maj[32], 0), 'from ' + pct(pS.get(), 0) + ' at N = 1'); put('b', pct(r.best[32], 0), 'verifier picks the top score'); put('p', pct(r.pass[32], 0), 'some sample is right');
      var xs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];
      ch({ x: { min: 1, max: 32, log: true, label: 'samples N (log)', ticks: [1, 2, 4, 8, 16, 32], fmt: String }, y: { min: 0, max: 1, label: 'accuracy', ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (v) { return pct(v, 0); } },
        legend: true, series: [{ pts: xs.map(function (x) { return [x, r.maj[x]]; }), cls: 's-a', label: 'majority vote' }, { pts: xs.map(function (x) { return [x, r.best[x]]; }), cls: 's-b', label: 'best-of-N (noisy verifier)' }, { pts: xs.map(function (x) { return [x, r.pass[x]]; }), cls: 's-c dash', label: 'pass@N (oracle)' }] });
    }
    draw();
  };

  // ---------------------------------------------------------- bpe
  LABS.bpe = function (root) {
    var f = frame(root, 'Train a BPE tokenizer', 'Count adjacent symbol pairs (weighted by word frequency), merge the most frequent, repeat. Ties go to the first pair seen. Then tokenize a new word with the learned merges.');
    var ta = h('textarea', { 'aria-label': 'training text' }); ta.value = 'low low low low low newer newer newer newest';
    f.controls.appendChild(h('label', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', { text: 'training text' })), ta));
    var mS = slider(f.controls, { label: 'merges', min: 0, max: 30, value: 7, onchange: draw });
    var wIn = h('input', { type: 'text', value: 'lower' });
    f.controls.appendChild(h('label', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', { text: 'encode a word' })), wIn));
    ta.addEventListener('input', draw); wIn.addEventListener('input', draw);
    var put = readouts(f.out, [{ id: 'v', k: 'vocabulary', hot: true }, { id: 'e', k: 'encoded word' }]);
    var enc = h('div', { class: 'tokens-out' }); f.out.appendChild(enc);
    var traceBox = h('div'); f.out.appendChild(traceBox);
    function draw() {
      var text = ta.value.toLowerCase(), r = bpeTrain(text, mS.get()), chars = {};
      text.replace(/\s+/g, '').split('').forEach(function (c) { chars[c] = 1; });
      put('v', Object.keys(chars).length + r.merges.length + ' symbols', Object.keys(chars).length + ' characters + ' + r.merges.length + ' merges');
      var e = bpeEncode(wIn.value.toLowerCase(), r.merges); put('e', e.length + ' tokens', '');
      enc.innerHTML = ''; e.forEach(function (t) { enc.appendChild(h('span', { class: 'tk new', text: t })); });
      traceBox.innerHTML = '';
      r.trace.slice(-8).forEach(function (t, i, arr) {
        var idx = r.trace.length - arr.length + i + 1;
        var line = h('div', { class: 'fmt-row' }, h('span', { class: 'fk', text: '#' + idx + ' ' + t.a + '+' + t.b }), h('span', { class: 'fv', text: '→ ' + t.a + t.b + ' (×' + t.count + ')   ' + t.corpus.slice(0, 4).map(function (c) { return c.sym.join(' ') + (c.f > 1 ? ' ×' + c.f : ''); }).join(' · ') }));
        traceBox.appendChild(line);
      });
    }
    draw();
  };

  // ---------------------------------------------------------- masks (attention patterns)
  LABS.masks = function (root) {
    var f = frame(root, 'Attention pattern explorer', 'Rows are queries, columns keys. Each pattern decides which pairs are computed — and therefore how much of the KV cache a layer must keep and read.');
    var kS = select(f.controls, { label: 'pattern', options: [['causal', 'Causal (decoder)'], ['bidirectional', 'Bidirectional (encoder)'], ['prefix', 'Prefix-LM'], ['sliding', 'Sliding window'], ['chunked', 'Chunked (Llama 4)'], ['sink', 'Sinks + window (StreamingLLM)'], ['dilated', 'Dilated'], ['nsa', 'Block-sparse (NSA)'], ['topk', 'Top-k (DeepSeek DSA)']], value: 'sliding', onchange: draw });
    var tS = slider(f.controls, { label: 'sequence length T', min: 8, max: 48, step: 4, value: 24, onchange: draw });
    var wS = slider(f.controls, { label: 'window W', min: 1, max: 16, value: 6, onchange: draw });
    var cS = slider(f.controls, { label: 'chunk / block size', min: 2, max: 12, value: 6, onchange: draw });
    var sS = slider(f.controls, { label: 'sinks / selected blocks', min: 1, max: 4, value: 2, onchange: draw });
    var kkS = slider(f.controls, { label: 'top-k keys', min: 1, max: 16, value: 6, onchange: draw });
    var put = readouts(f.out, [{ id: 'p', k: 'pairs computed', hot: true }, { id: 'r', k: 'last query reads' }, { id: 'c', k: 'KV the layer must keep' }]);
    var grid = h('div', { class: 'maskgrid' }); f.out.appendChild(grid);
    var note = h('div', { class: 'lab-note' }); f.out.appendChild(note);
    function draw() {
      var kind = kS.get(), T = tS.get(), o = { window: wS.get(), chunk: cS.get(), block: cS.get(), sinks: sS.get(), blocks: sS.get(), prefix: Math.round(T / 3), topk: kkS.get(), seed: 5 };
      var m = maskPattern(kind, T, o), st = maskStats(m);
      grid.innerHTML = ''; grid.style.gridTemplateColumns = 'repeat(' + T + ', 1fr)';
      var cls = ['', 'a', 's', 'w'];
      for (var i = 0; i < T; i++) for (var j = 0; j < T; j++) { var c = h('i'); if (m[i][j]) c.className = cls[m[i][j]]; grid.appendChild(c); }
      put('p', st.pairs.toLocaleString('en-US'), pct(st.frac, 0) + ' of full causal (' + st.causalPairs + ')');
      put('r', st.lastRow + ' of ' + T + ' keys', '');
      var keep = { causal: T, bidirectional: T, prefix: T, sliding: Math.min(T, o.window), chunked: ((T - 1) % o.chunk) + 1, sink: Math.min(T, o.sinks + o.window), dilated: Math.min(T, 2 * o.window), nsa: T, topk: T }[kind];
      put('c', keep + ' tokens', keep < T ? 'bounded — the cache stops growing' : 'all of them: grows with T');
      note.innerHTML = {
        causal: 'Every token attends to all earlier tokens: T(T+1)/2 pairs, and the cache keeps everything.',
        bidirectional: 'Encoders see the whole sequence in both directions; nothing is generated token by token, so there is no KV cache to grow.',
        prefix: 'The prompt (first third here) attends bidirectionally; generated tokens attend causally — T5-style prefix LMs and some VLM image tokens.',
        sliding: 'Each query sees the last W keys. The cache is a ring buffer of W tokens, and information travels further only by passing through layers (W per layer).',
        chunked: 'Tokens attend only within their chunk (Llama 4\'s local layers); interleaved global layers reconnect the chunks.',
        sink: 'Keeping the first few tokens plus a window stabilises streaming: softmax needs somewhere to dump attention (Chapter 4\'s attention sinks).',
        dilated: 'Skipping keys widens the receptive field for the same cost — used in older sparse transformers, rarely in 2026 LLMs.',
        nsa: 'Native sparse attention: a compressed summary picks the most relevant blocks, which are read in full alongside a local window. The cache keeps everything; each query reads a fraction.',
        topk: 'DeepSeek sparse attention (V3.2): a tiny FP8 "lightning indexer" scores all previous tokens and the main attention reads only the top k. Reads shrink from T to k; the cache does not.'
      }[kind];
    }
    draw();
  };

  // ---------------------------------------------------------- loop (adaptive depth)
  LABS.loop = function (root) {
    var f = frame(root, 'Adaptive depth: when should a token stop looping?', 'A toy recurrent block h ← c·tanh(W h) + e runs for each token; c sets how hard the token is (how slowly its state settles). A token exits when its update is smaller than ε of its state — the convergence-style exit of recurrent-depth models.');
    var eS = slider(f.controls, { label: 'exit threshold ε', min: -6, max: -1, step: 0.25, value: -3, map: function (v) { return Math.pow(10, v); }, fmt: function (v) { return v.toExponential(0); }, onchange: draw });
    var rS = slider(f.controls, { label: 'max loops r_max', min: 4, max: 48, step: 4, value: 32, onchange: draw });
    var dS = slider(f.controls, { label: 'hardest token (c_max)', min: 0.5, max: 0.98, step: 0.01, value: 0.92, fmt: function (v) { return v.toFixed(2); }, onchange: draw });
    var put = readouts(f.out, [{ id: 'm', k: 'mean loops per token', hot: true }, { id: 'c', k: 'compute vs always r_max' }, { id: 'd', k: 'effective depth (3-layer block)' }]);
    var grid = h('div', { class: 'loopgrid' }); f.out.appendChild(grid);
    var ch = chart(f.out, 200);
    function draw() {
      var r = loopSim({ tokens: 12, cMin: 0.15, cMax: dS.get(), rMax: rS.get(), eps: eS.get(), seed: 3 });
      put('m', r.meanLoops.toFixed(1), 'of ' + rS.get());
      put('c', pct(r.computeFraction, 0), 'parameters unchanged');
      var lo = Math.min.apply(null, r.tokens.map(function (t) { return t.loops; })), hi = Math.max.apply(null, r.tokens.map(function (t) { return t.loops; }));
      put('d', 3 * lo + '–' + 3 * hi + ' layers', 'from the same 3 layers of weights');
      grid.innerHTML = ''; grid.style.gridTemplateColumns = '5.5rem repeat(' + rS.get() + ', 1fr)';
      r.tokens.forEach(function (t, k) {
        grid.appendChild(h('span', { class: 'lg-k', text: 't' + (k + 1) + '  c ' + t.c.toFixed(2) }));
        for (var i = 0; i < rS.get(); i++) {
          var c = h('i');
          if (i < t.loops) { var v = clamp(-Math.log10(t.residuals[i]) / 6, 0, 1); c.style.background = 'color-mix(in srgb, var(--purple) ' + Math.round(90 - v * 70) + '%, var(--paper-2))'; if (i === t.loops - 1) c.className = 'exit'; }
          grid.appendChild(c);
        }
      });
      var pick = [0, 4, 8, 11];
      ch({ x: { min: 1, max: rS.get(), label: 'loop', ticks: [1, Math.round(rS.get() / 4), Math.round(rS.get() / 2), Math.round(3 * rS.get() / 4), rS.get()], fmt: String },
        y: { min: 1e-7, max: 10, log: true, label: 'update', ticks: [1e-6, 1e-3, 1], fmt: function (v) { return v === 1 ? '1' : '1e' + Math.round(Math.log10(v)); } },
        legend: true,
        series: pick.map(function (k, i) { var t = r.tokens[k]; return { pts: t.residuals.map(function (v, j) { return [j + 1, Math.max(v, 1e-7)]; }), cls: ['s-c', 's-b', 's-d', 's-a'][i], label: 'token ' + (k + 1) + ' (c=' + t.c.toFixed(2) + ')' }; }),
        hlines: [{ y: eS.get(), label: 'ε' }] });
    }
    draw();
  };

  // shared with lm-figs.js, which registers labs that draw figures
  LABS.__frame = { frame: frame, slider: slider, select: select, check: check, readouts: readouts, barList: barList, chart: chart };
  window.LMLabs = LABS;

  // ============================================================ boot
  function boot() {
    Array.prototype.forEach.call(document.querySelectorAll('.lab[data-lab]'), function (el) {
      var fn = LABS[el.dataset.lab];
      if (typeof fn !== 'function') return;
      try { fn(el); } catch (e) { el.innerHTML = '<div class="lab-head"><span class="lab-badge">Lab</span><span class="lab-title">This lab failed to load</span></div><p class="lab-desc">' + String(e && e.message) + '</p>'; if (window.console) console.error(e); }
    });
  }
  // wait for DOMContentLoaded so that lm-figs.js (loaded after this file) can register its labs first
  if (document.readyState === 'complete') boot(); else document.addEventListener('DOMContentLoaded', boot);
})();
