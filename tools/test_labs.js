#!/usr/bin/env node
/* Checks the labs' arithmetic against numbers quoted in the chapters.
   Run:  node tools/test_labs.js        (no dependencies) */
'use strict';
var L = require('../assets/js/lm-labs.js');
var fails = 0, n = 0;
function near(name, got, want, tol) {
  n++;
  var ok = Math.abs(got - want) <= (tol === undefined ? 1e-9 : tol) * (Math.abs(want) || 1);
  if (!ok) { fails++; console.log('FAIL ' + name + ': got ' + got + ', want ' + want); }
}
function eq(name, got, want) {
  n++;
  var a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { fails++; console.log('FAIL ' + name + ': got ' + a + ', want ' + b); }
}
function ok(name, cond) { n++; if (!cond) { fails++; console.log('FAIL ' + name); } }

// ---- Ch 3 / App. C: parameter counts
var p8 = L.countParams(L.MODELS['llama31-8b']);
eq('Llama 3.1 8B total (exact)', p8.total, 8030261248);
near('Llama 3.1 70B total', L.countParams(L.MODELS['llama31-70b']).total, 70.55e9, 0.002);
near('Llama 3.1 405B total', L.countParams(L.MODELS['llama31-405b']).total, 405.85e9, 0.002);
near('Qwen3-8B total', L.countParams(L.MODELS['qwen3-8b']).total, 8.19e9, 0.002);
near('Qwen3-0.6B total (tied)', L.countParams(L.MODELS['qwen3-0.6b']).total, 0.596e9, 0.005);
var q30 = L.countParams(L.MODELS['qwen3-30b-a3b']);
near('Qwen3-30B-A3B total', q30.total, 30.5e9, 0.005);
near('Qwen3-30B-A3B active', q30.active, 3.35e9, 0.01);
near('Mixtral total', L.countParams(L.MODELS['mixtral-8x7b']).total, 46.7e9, 0.003);
near('Mixtral active', L.countParams(L.MODELS['mixtral-8x7b']).active, 12.9e9, 0.005);

// ---- Ch 5: KV bytes per token
var K = L.KV_PRESETS;
eq('KV Llama 8B / token', L.kvBytes(K['llama31-8b'], 1), 131072);
eq('KV Llama 8B as MHA / token', L.kvBytes(K['llama31-8b-mha'], 1), 524288);
eq('KV Llama 70B / token', L.kvBytes(K['llama31-70b'], 1), 327680);
eq('KV DeepSeek-V3 MLA / token', L.kvBytes(K['deepseek-v3'], 1), 70272);
eq('KV Qwen3-235B / token', L.kvBytes(K['qwen3-235b'], 1), 192512);
eq('KV gpt-oss-120b at 128K', L.kvBytes(K['gpt-oss-120b'], 131072), 36864 * 131072 + 18 * 2048 * 128);
near('KV Llama 8B at 128K = 16 GiB', L.kvBytes(K['llama31-8b'], 131072) / Math.pow(2, 30), 16, 1e-12);

// ---- Ch 10: compute
near('6ND Llama 3.1 405B 15.6T tokens', L.trainFlops(405e9, 15.6e12), 3.79e25, 0.003);
near('GPU-hours at 40% MFU on H100 (989 TF)', L.gpuHours(3.79e25, 989, 0.4), 26.6e6, 0.01);

// ---- Ch 13: fine-tuning memory (Llama 3.1 8B, 2K tokens, batch 1, checkpointing, +1.5 GB overhead)
function ft(method) {
  var c = L.MODELS['llama31-8b'], p = L.countParams(c), P = 16 * L.loraPerRank(c, ['q', 'k', 'v', 'o', 'gate', 'up', 'down']);
  return L.finetuneMem({ N: p.total, embParams: p.emb + p.head, method: method, loraP: method === 'full' ? 0 : P, seq: 2048, batch: 1, d: c.d, L: c.L, V: c.V, ckpt: true, fusedCE: false, overhead: 1.5e9 }).total / 1e9;
}
var full = ft('full'), lora = ft('lora'), qlora = ft('qlora');
ok('full FT ≈ 130 GB (got ' + full.toFixed(1) + ')', full > 125 && full < 135);
ok('LoRA ≈ 20–24 GB (got ' + lora.toFixed(1) + ')', lora > 18 && lora < 24);
ok('QLoRA ≈ 9–12 GB (got ' + qlora.toFixed(1) + ')', qlora > 8.5 && qlora < 12);

// ---- Ch 14: LoRA parameter counts
eq('LoRA all-linear per unit rank (8B) = 32 × 81,920', L.loraPerRank(L.MODELS['llama31-8b'], ['q', 'k', 'v', 'o', 'gate', 'up', 'down']), 32 * 81920);
eq('LoRA r=16 all-linear (8B)', 16 * L.loraPerRank(L.MODELS['llama31-8b'], ['q', 'k', 'v', 'o', 'gate', 'up', 'down']), 41943040);
eq('LoRA r=8 q,v (8B)', 8 * L.loraPerRank(L.MODELS['llama31-8b'], ['q', 'v']), 3407872);
near('energy captured is 1 at full rank', L.energyCaptured(64, 0.8, 64), 1, 1e-12);

// ---- Ch 15: GRPO
eq('advantages', L.groupAdvantages([1, 0, 1, 0]).map(function (x) { return +x.toFixed(6); }), [1, -1, 1, -1]);
eq('all-equal group has zero advantages', L.groupAdvantages([1, 1, 1, 1]), [0, 0, 0, 0]);
near('zero-gradient fraction p=0.95, G=8', L.zeroGradFraction(0.95, 8), 0.6634, 1e-3);
near('zero-gradient fraction p=0.5, G=8', L.zeroGradFraction(0.5, 8), 0.0078125, 1e-9);

// ---- Ch 16: sampling and speculative decoding
var lg = [2, 1, 0.5, -1];
near('greedy is one-hot', L.shapeLogits(lg, { T: 0, topK: 0, topP: 1, minP: 0 })[0], 1);
eq('top-k 2 keeps two', L.shapeLogits(lg, { T: 1, topK: 2, topP: 1, minP: 0 }).filter(function (x) { return x > 0; }).length, 2);
var sm = L.softmax(lg);
eq('top-p keeps smallest prefix reaching p', L.shapeLogits(lg, { T: 1, topK: 0, topP: sm[0] + 1e-9, minP: 0 }).filter(function (x) { return x > 0; }).length, 2);
eq('top-p exactly at first token keeps one', L.shapeLogits(lg, { T: 1, topK: 0, topP: sm[0] - 1e-9, minP: 0 }).filter(function (x) { return x > 0; }).length, 1);
eq('min-p 0.3 keeps tokens ≥ 0.3 × max', L.shapeLogits(lg, { T: 1, topK: 0, topP: 1, minP: 0.3 }).filter(function (x) { return x > 0; }).length, 2);
near('E[tokens] α=0.7 K=5', L.specTokens(0.7, 5), 2.94, 0.002);
near('E[tokens] α=0.9 K=5', L.specTokens(0.9, 5), 4.69, 0.002);
near('E[tokens] α=1 K=5', L.specTokens(1, 5), 6);
near('speed-up α=0.8 K=4 c=0.05', L.specSpeedup(0.8, 4, 0.05), 3.3616 / 1.2, 1e-6);

// ---- Ch 18: paged allocation never admits fewer requests than contiguous
var lens = L.requestStream(400, 5, 1200, 8192), r = L.allocate(lens, 1 << 19, 16, 8192);
ok('paged admits ≥ contiguous', r.paged.n >= r.contiguous.n);
ok('paged waste < 2% with 16-token blocks', r.paged.waste < 0.02);
ok('contiguous waste > 50% at 8K reservation', r.contiguous.waste > 0.5);

// ---- Ch 19: number formats
var F = L.FORMATS;
eq('BF16(100000)', L.fpEncode(F.bf16, 100000).value, 99840);
eq('FP16(100000) overflows', L.fpEncode(F.fp16, 100000).value, Infinity);
eq('FP16 max', L.fpEncode(F.fp16, 65504).value, 65504);
eq('E4M3 max 448', L.fpEncode(F.e4m3, 1000).value, 448);
eq('E5M2 max 57344', L.fpEncode(F.e5m2, 57344).value, 57344);
eq('FP4 grid', [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7].map(function (x) { return L.fpEncode(F.e2m1, x).value; }), [0, 0, 0.5, 1, 1, 1, 1.5, 2, 2, 2, 3, 4, 4, 4, 6, 6]);
eq('FP6 E3M2 max', L.fpEncode(F.e3m2, 1e9).value, 28);
eq('FP6 E2M3 max', L.fpEncode(F.e2m3, 1e9).value, 7.5);
eq('BF16(0.3) bits', L.fpEncode(F.bf16, 0.3).code.toString(2).padStart(16, '0'), '0011111010011010');
eq('FP32(0.1) value', L.fpEncode(F.fp32, 0.1).value, 0.10000000149011612);
eq('FP16 smallest subnormal', L.fpDecode(F.fp16, 1), Math.pow(2, -24));
eq('E4M3 0x7F is NaN', isNaN(L.fpDecode(F.e4m3, 0x7f)), true);
eq('ties to even: BF16(1 + 2^-8) → 1', L.fpEncode(F.bf16, 1 + Math.pow(2, -8)).value, 1);
eq('ties to even: BF16(1 + 3·2^-8) → 1 + 2^-6', L.fpEncode(F.bf16, 1 + 3 * Math.pow(2, -8)).value, 1 + Math.pow(2, -6));

// ---- Ch 20: quantization
var w = L.weightRow(4096, 9, 0.002, 30);
var eT = L.relError(w, L.quantize(w, 4, 'sym', 0)), e128 = L.relError(w, L.quantize(w, 4, 'sym', 128)), e32 = L.relError(w, L.quantize(w, 4, 'sym', 32));
ok('smaller groups help with outliers (' + eT.toFixed(3) + ' > ' + e128.toFixed(3) + ' > ' + e32.toFixed(3) + ')', eT > e128 && e128 > e32);
ok('INT8 per-tensor error small', L.relError(w, L.quantize(w, 8, 'sym', 0)) < 0.1);
eq('symmetric INT4 of exact grid is lossless', L.quantize([-7, -1, 0, 3, 7], 4, 'sym', 0), [-7, -1, 0, 3, 7]);

// ---- Ch 22/24: ceilings and roofline
near('decode ceiling: 8B FP16 on H100', L.decodeCeiling(16.06e9, 3.35), 208.6, 0.005);
near('H100 BF16 ridge', 989 / 3.35, 295.2, 0.001);
near('roofline below ridge', L.attainable(10, 989, 3.35), 33.5);
near('roofline above ridge', L.attainable(1000, 989, 3.35), 989);
near('GEMV intensity ≈ 1 FLOP/B in BF16', L.gemmIntensity(1, 4096, 4096, 2), 1, 0.001);

// ---- Ch 26: parallelism
near('no sharding: 16 B/param', L.parallelMem({ N: 7e9, tp: 1, pp: 1, dp: 8, zero: 0 }).total, 112e9);
near('ZeRO-3 over 8', L.parallelMem({ N: 7e9, tp: 1, pp: 1, dp: 8, zero: 3 }).total, 14e9);
near('ZeRO-1 over 8', L.parallelMem({ N: 7e9, tp: 1, pp: 1, dp: 8, zero: 1 }).total, 7e9 * (4 + 12 / 8));

// ---- Ch 31: serving cost
near('$/M at $2.50/h, 2,500 tok/s', L.costPerMillion(2.5, 2500), 0.2778, 1e-3);
near('$/M at $2.50/h, 100 tok/s', L.costPerMillion(2.5, 100), 6.944, 1e-3);
var sp = { weightBytes: 8e9, kvBytesPerSeq: 0, bw: 3.35, peak: 1979, activeN: 8e9, effBw: 1, effCmp: 1 };
near('batch-1 decode = bandwidth ceiling', L.servePoint(sp, 1).perStream, 3.35e12 / 8e9, 1e-9);
ok('batching raises total throughput', L.servePoint(sp, 64).total > 30 * L.servePoint(sp, 1).total);

// ---- Ch 32: scaling (Epoch AI refit)
var o = L.chinOptimal(1e23);
near('N* at 1e23', o.N, 29.4e9, 0.02);
near('D* at 1e23', o.D, 0.57e12, 0.02);
near('6·N*·D* = C', 6 * o.N * o.D, 1e23, 1e-9);
ok('optimum is a minimum along the isoFLOP', L.chinLoss(o.N, o.D) < L.chinLoss(o.N * 1.5, 1e23 / (6 * o.N * 1.5)) && L.chinLoss(o.N, o.D) < L.chinLoss(o.N / 1.5, 1e23 / (6 * o.N / 1.5)));

// ---- Ch 33: routing simulation
function route(method, E, extra) { var o = { E: E, k: 2, tokens: 256, steps: 60, method: method, capacity: 0, eta: 0.04, alpha: 0.3, gamma: 0.15, seed: 7 }; for (var key in extra || {}) o[key] = extra[key]; return L.simulateRouting(o); }
[16, 32, 64].forEach(function (E) {
  var none = route('none', E), aux = route('aux', E), bias = route('bias', E);
  ok('E=' + E + ': no balancing collapses (max/mean ' + none.imbalance[59].toFixed(1) + ', ' + none.dead + ' dead)', none.imbalance[59] >= E / 4 && none.dead >= E / 2);
  ok('E=' + E + ': aux loss keeps max/mean < 3 (' + aux.imbalance[59].toFixed(2) + ')', aux.imbalance[59] < 3);
  ok('E=' + E + ': bias keeps max/mean < 3 with no dead experts (' + bias.imbalance[59].toFixed(2) + ')', bias.imbalance[59] < 3 && bias.dead === 0);
  eq('E=' + E + ': dropless drops nothing', none.dropped, 0);
});
ok('capacity 1.25 drops tokens when collapsing', route('none', 16, { capacity: 1.25 }).dropped > 0.1);

// ---- Ch 34: fixed state recall
// chapter (NumPy, d = 64): 16 pairs 0.43/0.31, 64 pairs 1.00/0.76, 256 pairs 1.97/1.22 (linear/delta)
near('linear error at n = d = 64', L.recallError(64, 64, 'linear', 1), 1.00, 0.08);
near('delta error at n = d = 64', L.recallError(64, 64, 'delta', 1), 0.76, 0.08);
ok('delta beats linear at every n', [16, 64, 256].every(function (n) { return L.recallError(n, 64, 'delta', 1) < L.recallError(n, 64, 'linear', 1); }));
ok('softmax cache stays exact', L.recallError(256, 64, 'softmax', 1) < 0.01);

// ---- Ch 35: voting
var v = L.votingCurve({ p: 0.4, concentration: 1, noise: 0.6, maxN: 32, trials: 1500, seed: 11 });
near('majority@1 ≈ p', v.maj[1], 0.4, 0.1);
ok('majority improves with N when the right answer is the plurality', v.maj[32] > v.maj[1] + 0.2);
ok('pass@N ≥ majority', v.pass[32] >= v.maj[32]);

// ---- App. B: BPE
var b = L.bpeTrain('low low low low low newer newer newer newest', 7);
eq('BPE merges', b.merges, [['l', 'o'], ['lo', 'w'], ['n', 'e'], ['ne', 'w'], ['new', 'e'], ['newe', 'r'], ['newe', 's']]);
eq('encode "lower"', L.bpeEncode('lower', b.merges), ['low', 'e', 'r']);
eq('encode "newest"', L.bpeEncode('newest', b.merges), ['newes', 't']);

// ---- Ch 1: n-gram
var m = L.trainNgram(L.tokenizeWords(L.CORPUS));
var nx = L.ngramNext(m, ['the', 'kv']);
eq('trigram context found', nx.order, 3);
eq('"the kv" → "cache"', nx.dist[0].tok, 'cache');
near('distribution sums to 1', nx.dist.reduce(function (s, d) { return s + d.p; }, 0), 1, 1e-12);
var sh = L.reshapeDist(L.ngramNext(m, ['the']).dist, 0.5, 3);
eq('top-3 after reshaping', sh.filter(function (d) { return d.p > 0; }).length, 3);

// ---- Ch 2: RoPE relative position
var q = [1, 0.3, 0.8, -0.2, 0.5, 0.6, 0.9, 0.1], k = [0.9, 0.2, 0.7, -0.1, 0.6, 0.5, 1.0, 0.2];
near('RoPE: score depends only on m − n', L.ropeScore(q, k, 12, 9, 10000), L.ropeScore(q, k, 103, 100, 10000), 1e-9);
near('RoPE: rotation preserves norm', Math.hypot.apply(null, L.ropeRotate(q, 37, 10000)), Math.hypot.apply(null, q), 1e-12);
near('RoPE: θ_0 = 1', L.ropeFreq(0, 128, 500000), 1);

console.log((n - fails) + '/' + n + ' lab checks passed');
process.exit(fails ? 1 : 0);
