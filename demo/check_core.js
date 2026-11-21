// Check of the browser core (docs/assets/bm-core.js) against the Python reference (demo/reference.json) on the demo data.
// Usage (repository root):  python -m demo.reference && node demo/check_core.js
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const BM = require(path.join(root, "docs", "assets", "bm-core.js"));
global.window = {};
require(path.join(root, "docs", "assets", "demo-data.js"));
const D = global.window.BM_DEMO_DATA;
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json"), "utf8"));

const d = D.meta.dim;
const C = D.meta.classes.length;
function decode(part) {
  const raw = Buffer.from(part.x, "base64");
  const n = raw.length / d;
  const X = new Float32Array(n * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = raw.readInt8(i * d + j) * D.scale[j];
  return { X, n, y: Uint8Array.from(Buffer.from(part.y, "base64")), z: Uint8Array.from(Buffer.from(part.z, "base64")) };
}
const tr = decode(D.train);
const te = decode(D.test);

let failures = 0;
function compare(name, js, py, tolAcc, tolGap) {
  const dAcc = Math.abs(js.acc - py.acc);
  const dGap = Math.abs(js.gapRms - py.gap_rms);
  const ok = dAcc <= tolAcc && dGap <= tolGap;
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(26)} acc ${js.acc.toFixed(4)} (py ${py.acc.toFixed(4)})  gap ${js.gapRms.toFixed(4)} (py ${py.gap_rms.toFixed(4)})`);
}
const evaluate = (model, X) => BM.recallSummary(te.y, te.z, BM.predict(model, X || te.X, te.n), te.n, C);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

// Original Model
const base = BM.fitSoftmax(tr.X, tr.n, d, tr.y, C, { l2: 1e-3 });
compare("original", evaluate(base), ref.none, 0.002, 0.003);

// Minimal Plan
const { mu, cnt } = BM.cellStats(tr.X, tr.n, d, tr.y, tr.z, C);
const mp = BM.minimalPlan(cnt, C);
const additions = Array.from({ length: C }, (_, c) => [Math.round(mp.plan[c * 2]), Math.round(mp.plan[c * 2 + 1])]);
const planOk = JSON.stringify(additions) === JSON.stringify(ref.plan.additions) && Math.abs(mp.q - ref.plan.q) < 1e-6;
if (!planOk) failures++;
console.log(`${planOk ? "ok  " : "FAIL"} minimal plan               q ${mp.q.toFixed(6)} (py ${ref.plan.q.toFixed(6)}), ${additions.flat().reduce((a, b) => a + b, 0)} additions (py ${ref.plan.total})`);

// BME: edited embeddings, retrained and frozen models
for (const [name, opts] of [["bme_oblique", { oblique: true }], ["bme_orth", { oblique: false }], ["bme_oblique_k1", { oblique: true, k: 1 }], ["bme_oblique_k3", { oblique: true, k: 3 }]]) {
  const er = BM.bmeEraser(tr.X, tr.n, d, tr.y, tr.z, C, opts);
  const E = BM.applyEraser(er, tr.X, tr.n);
  const Et = BM.applyEraser(er, te.X, te.n);
  let diff = 0;
  ref[name].edited_test_head.forEach((row, i) => row.forEach((v, j) => { diff = Math.max(diff, Math.abs(Et[i * d + j] - v)); }));
  const editOk = diff < 1e-3;
  if (!editOk) failures++;
  console.log(`${editOk ? "ok  " : "FAIL"} ${(name + " edit").padEnd(26)} max |edited - py| ${diff.toExponential(2)}`);
  compare(name + " retrained", evaluate(BM.fitSoftmax(E, tr.n, d, tr.y, C, { l2: 1e-3 }), Et), ref[name].retrained, 0.002, 0.003);
  compare(name + " frozen", evaluate(base, Et), ref[name].frozen, 0.002, 0.003);
}

// BMA: mean over five draws of the sources (the two generators differ)
const runs = [];
for (let seed = 1; seed <= 5; seed++) {
  const cp = BM.bmaCopies(tr.X, d, tr.y, mu, BM.planSources(tr.y, tr.z, tr.n, C, mp.plan, BM.makeRng(seed)), 1);
  const X = new Float32Array((tr.n + cp.m) * d);
  X.set(tr.X);
  X.set(cp.Xc, tr.n * d);
  const y = new Uint8Array(tr.n + cp.m);
  y.set(tr.y);
  y.set(cp.yc, tr.n);
  runs.push(evaluate(BM.fitSoftmax(X, tr.n + cp.m, d, y, C, { l2: 1e-3 })));
}
compare("bma (mean of 5 draws)", { acc: mean(runs.map((r) => r.acc)), gapRms: mean(runs.map((r) => r.gapRms)) },
  { acc: mean(ref.bma.acc), gap_rms: mean(ref.bma.gap_rms) }, 0.005, 0.01);

console.log(failures ? `${failures} check(s) failed` : "all checks passed");
process.exit(failures ? 1 : 0);
