/* Numerical core of the demos: seeded RNG, softmax regression (L-BFGS), conditional bias matrix,
   BME erasers, BMA copies on the minimal plan, recall metrics.  Binary groups (z in {0, 1}).
   Matrices are flat typed arrays in row-major order. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.BM = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- random
  // mulberry32 with Box-Muller normals and Fisher-Yates shuffles
  function makeRng(seed) {
    let s = seed >>> 0;
    let spare = null;
    function uniform() {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    function normal() {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      let u = 0;
      while (u === 0) u = uniform();
      const r = Math.sqrt(-2 * Math.log(u));
      const a = 2 * Math.PI * uniform();
      spare = r * Math.sin(a);
      return r * Math.cos(a);
    }
    function int(n) {
      return Math.floor(uniform() * n);
    }
    function shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = int(i + 1);
        const t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
      return arr;
    }
    return { uniform, normal, int, shuffle };
  }

  // ------------------------------------------------------------ statistics
  // Means (C*2, d) and counts (C*2) of the (class, group) cells; cell index = 2 * class + group
  function cellStats(X, n, d, y, z, C) {
    const mu = new Float64Array(C * 2 * d);
    const cnt = new Float64Array(C * 2);
    for (let i = 0; i < n; i++) {
      const c = y[i] * 2 + z[i];
      const o = c * d;
      const r = i * d;
      cnt[c]++;
      for (let j = 0; j < d; j++) mu[o + j] += X[r + j];
    }
    for (let c = 0; c < C * 2; c++) {
      if (cnt[c] > 0) for (let j = 0; j < d; j++) mu[c * d + j] /= cnt[c];
    }
    return { mu, cnt };
  }

  // Mean (d) and covariance (d, d)
  function covariance(X, n, d) {
    const mean = new Float64Array(d);
    const S = new Float64Array(d * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    const row = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) row[j] = X[i * d + j] - mean[j];
      for (let a = 0; a < d; a++) {
        const v = row[a];
        const o = a * d;
        for (let b = a; b < d; b++) S[o + b] += v * row[b];
      }
    }
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) {
        S[a * d + b] /= n;
        S[b * d + a] = S[a * d + b];
      }
    }
    return { mean, S };
  }

  // ---------------------------------------------------------- linear algebra
  // Symmetric eigendecomposition (cyclic Jacobi): eigenvalues w, eigenvectors in the columns of V
  function symEig(A, d) {
    const a = Float64Array.from(A);
    const V = new Float64Array(d * d);
    for (let i = 0; i < d; i++) V[i * d + i] = 1;
    let total = 0;
    for (let i = 0; i < d * d; i++) total += a[i] * a[i];
    for (let sweep = 0; sweep < 80; sweep++) {
      let off = 0;
      for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += a[p * d + q] * a[p * d + q];
      if (off <= 1e-26 * total) break;
      for (let p = 0; p < d - 1; p++) {
        for (let q = p + 1; q < d; q++) {
          const apq = a[p * d + q];
          if (apq === 0) continue;
          const theta = (a[q * d + q] - a[p * d + p]) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1);
          const s = t * c;
          for (let k = 0; k < d; k++) {
            const akp = a[k * d + p];
            const akq = a[k * d + q];
            a[k * d + p] = c * akp - s * akq;
            a[k * d + q] = s * akp + c * akq;
          }
          for (let k = 0; k < d; k++) {
            const apk = a[p * d + k];
            const aqk = a[q * d + k];
            a[p * d + k] = c * apk - s * aqk;
            a[q * d + k] = s * apk + c * aqk;
          }
          for (let k = 0; k < d; k++) {
            const vkp = V[k * d + p];
            const vkq = V[k * d + q];
            V[k * d + p] = c * vkp - s * vkq;
            V[k * d + q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const w = new Float64Array(d);
    for (let i = 0; i < d; i++) w[i] = a[i * d + i];
    return { w, V };
  }

  // Symmetric PSD matrix power S^p (pseudo-inverse sense for p < 0)
  function symPower(S, d, p, rtol) {
    const { w, V } = symEig(S, d);
    let wmax = 0;
    for (let i = 0; i < d; i++) wmax = Math.max(wmax, w[i]);
    const out = new Float64Array(d * d);
    for (let k = 0; k < d; k++) {
      if (w[k] <= (rtol || 1e-10) * wmax) continue;
      const f = Math.pow(w[k], p);
      for (let a = 0; a < d; a++) {
        const va = V[a * d + k] * f;
        for (let b = 0; b < d; b++) out[a * d + b] += va * V[b * d + k];
      }
    }
    return out;
  }

  // Product of A (ra, ca) and B (ca, cb)
  function matmul(A, ra, ca, B, cb) {
    const out = new Float64Array(ra * cb);
    for (let i = 0; i < ra; i++) {
      for (let k = 0; k < ca; k++) {
        const v = A[i * ca + k];
        if (v === 0) continue;
        for (let j = 0; j < cb; j++) out[i * cb + j] += v * B[k * cb + j];
      }
    }
    return out;
  }

  // Left singular vectors U (d, rank) and singular values s of M (d, r), from the Gram matrix M^T M
  function leftSingular(M, d, r) {
    const G = new Float64Array(r * r);
    for (let a = 0; a < r; a++) {
      for (let b = a; b < r; b++) {
        let v = 0;
        for (let i = 0; i < d; i++) v += M[i * r + a] * M[i * r + b];
        G[a * r + b] = v;
        G[b * r + a] = v;
      }
    }
    const { w, V } = symEig(G, r);
    const order = Array.from({ length: r }, (_, i) => i).sort((i, j) => w[j] - w[i]);
    const s0 = Math.sqrt(Math.max(w[order[0]], 0));
    const keep = order.filter((i) => Math.sqrt(Math.max(w[i], 0)) > 1e-6 * s0);  // Gram route: looser rank tolerance
    const U = new Float64Array(d * keep.length);
    const s = new Float64Array(keep.length);
    keep.forEach((col, j) => {
      s[j] = Math.sqrt(w[col]);
      for (let i = 0; i < d; i++) {
        let v = 0;
        for (let b = 0; b < r; b++) v += M[i * r + b] * V[b * r + col];
        U[i * keep.length + j] = v / s[j];
      }
    });
    return { U, s, rank: keep.length };
  }

  // ------------------------------------------------------------ bias matrix
  // Conditional bias matrix for two groups: columns sqrt(n_y0 n_y1 / (n_y n)) (mu_y1 - mu_y0), shape (d, C).
  // Same B B^T as the general definition with one column sqrt(n_zy / n) (mu_zy - mu_y) per (class, group) pair.
  function biasMatrix(mu, cnt, C, d, n) {
    const B = new Float64Array(d * C);
    for (let c = 0; c < C; c++) {
      const n0 = cnt[c * 2];
      const n1 = cnt[c * 2 + 1];
      const f = n0 + n1 > 0 ? Math.sqrt((n0 * n1) / (n0 + n1) / n) : 0;
      for (let j = 0; j < d; j++) B[j * C + c] = f * (mu[(c * 2 + 1) * d + j] - mu[c * 2 * d + j]);
    }
    return B;
  }

  // ---------------------------------------------------------------- BME
  // Eraser x' = x - L R^T (x - mean) removing the top-k directions of B.
  // oblique = false: orthogonal projection (L = R = U_k).  oblique = true: projection in the coordinates
  // whitened by the covariance (L = S^{1/2} Q_k, R = S^{-1/2} Q_k, Q = left singular vectors of S^{-1/2} B).
  function bmeEraser(X, n, d, y, z, C, opts) {
    const { mu, cnt } = cellStats(X, n, d, y, z, C);
    const B = biasMatrix(mu, cnt, C, d, n);
    const { mean, S } = covariance(X, n, d);
    let L;
    let R;
    let sv;
    if (opts && opts.oblique) {
      const W = symPower(S, d, -0.5);
      const Wp = symPower(S, d, 0.5);
      const { U, s, rank } = leftSingular(matmul(W, d, d, B, C), d, C);
      L = { M: matmul(Wp, d, d, U, rank), cols: rank };
      R = { M: matmul(W, d, d, U, rank), cols: rank };
      sv = s;
    } else {
      const { U, s, rank } = leftSingular(B, d, C);
      L = { M: U, cols: rank };
      R = L;
      sv = s;
    }
    const k = Math.min(opts && opts.k != null ? opts.k : L.cols, L.cols);
    return { L: L.M, R: R.M, cols: L.cols, k, d, mean, s: sv };
  }

  // Edited copy of X (n, d); k: number of directions actually removed (default: the eraser's k);
  // strength: share of their component that is removed (default 1 = hard edit, below 1 = soft edit)
  function applyEraser(er, X, n, k, strength) {
    const share = strength == null ? 1 : strength;
    const d = er.d;
    const kk = Math.min(k == null ? er.k : k, er.cols);
    const out = new Float32Array(n * d);
    const coef = new Float64Array(kk);
    for (let i = 0; i < n; i++) {
      const r = i * d;
      coef.fill(0);
      for (let a = 0; a < d; a++) {
        const v = X[r + a] - er.mean[a];
        const o = a * er.cols;
        for (let j = 0; j < kk; j++) coef[j] += er.R[o + j] * v;
      }
      for (let a = 0; a < d; a++) {
        let v = X[r + a];
        const o = a * er.cols;
        for (let j = 0; j < kk; j++) v -= share * er.L[o + j] * coef[j];
        out[r + a] = v;
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- BMA
  // Fewest additions that make class and group independent (two groups): target share q of group 1 in every class.
  // total(q) = sum_y max(n_y1 / q, n_y0 / (1 - q)) is convex and piecewise smooth: candidates are the kinks
  // q = n_y1 / n_y and the stationary point of each piece.  Returns real-valued additions plan (C*2), q, total.
  function minimalPlan(cnt, C) {
    const total = (q) => {
      let t = 0;
      for (let c = 0; c < C; c++) t += Math.max(cnt[c * 2 + 1] / q, cnt[c * 2] / (1 - q));
      return t;
    };
    const kinks = [];
    for (let c = 0; c < C; c++) {
      const p = cnt[c * 2 + 1] / (cnt[c * 2] + cnt[c * 2 + 1]);
      if (p > 0 && p < 1) kinks.push(p);
    }
    kinks.sort((a, b) => a - b);
    const cand = kinks.slice();
    const edges = [1e-9].concat(kinks, [1 - 1e-9]);
    for (let e = 0; e + 1 < edges.length; e++) {
      const mid = 0.5 * (edges[e] + edges[e + 1]);
      let A = 0;
      let Bq = 0;
      for (let c = 0; c < C; c++) {
        if (cnt[c * 2 + 1] / mid >= cnt[c * 2] / (1 - mid)) A += cnt[c * 2 + 1];
        else Bq += cnt[c * 2];
      }
      if (A > 0 && Bq > 0) {
        const q = Math.sqrt(A) / (Math.sqrt(A) + Math.sqrt(Bq));
        if (q > edges[e] && q < edges[e + 1]) cand.push(q);
      }
    }
    let q = cand[0];
    for (const c of cand) if (total(c) < total(q)) q = c;
    const plan = new Float64Array(C * 2);
    for (let c = 0; c < C; c++) {
      const m = Math.max(cnt[c * 2 + 1] / q, cnt[c * 2] / (1 - q));
      plan[c * 2] = m * (1 - q) - cnt[c * 2];
      plan[c * 2 + 1] = m * q - cnt[c * 2 + 1];
    }
    return { plan, q, total: total(q) };
  }

  // Source rows of the additions of every cell: rows of the other group in the same class, in a fixed random
  // order (distinct rows first, repeats only for the excess), so a fraction of the plan is a prefix of the full plan.
  function planSources(y, z, n, C, plan, rng) {
    const pools = Array.from({ length: C * 2 }, () => []);
    for (let i = 0; i < n; i++) pools[y[i] * 2 + z[i]].push(i);
    const out = [];
    for (let c = 0; c < C; c++) {
      for (let g = 0; g < 2; g++) {
        const need = Math.round(plan[c * 2 + g]);
        const pool = pools[c * 2 + (1 - g)].slice();
        const src = [];
        if (need > 0 && pool.length > 0) {
          rng.shuffle(pool);
          for (let i = 0; i < Math.min(need, pool.length); i++) src.push(pool[i]);
          for (let i = pool.length; i < need; i++) src.push(pool[rng.int(pool.length)]);
        }
        out.push(src);
      }
    }
    return out;
  }

  // Mean-shift copies x + mu_{z,y} - mu_{z',y} for the first `frac` of every cell's sources
  function bmaCopies(X, d, y, mu, sources, frac) {
    const src = [];
    const tgt = [];
    sources.forEach((list, cell) => {
      const take = Math.round(list.length * frac);
      for (let i = 0; i < take; i++) {
        src.push(list[i]);
        tgt.push(cell);
      }
    });
    const m = src.length;
    const Xc = new Float32Array(m * d);
    const yc = new Uint8Array(m);
    const zc = new Uint8Array(m);
    for (let i = 0; i < m; i++) {
      const cell = tgt[i];
      const c = cell >> 1;
      const g = cell & 1;
      const to = cell * d;
      const from = (c * 2 + (1 - g)) * d;
      const r = src[i] * d;
      for (let j = 0; j < d; j++) Xc[i * d + j] = X[r + j] + mu[to + j] - mu[from + j];
      yc[i] = y[src[i]];
      zc[i] = g;
    }
    return { Xc, yc, zc, src: Int32Array.from(src), m };
  }

  // ------------------------------------------------------ softmax regression
  // Multinomial logistic regression on standardized inputs with an L2 penalty on W, trained by L-BFGS.
  // step(ms) runs iterations for about ms milliseconds, so a page can train without blocking.
  function makeTrainer(X, n, d, y, C, opts) {
    const o = Object.assign({ l2: 1e-3, maxIter: 400, gtol: 1e-5, ftol: 1e-10, history: 10, weight: null, init: null, standardize: true }, opts || {});
    const mean = new Float64Array(d);
    const std = new Float64Array(d).fill(1);
    if (o.standardize) {  // off: penalty in the raw coordinates (minimum-norm weights for degenerate inputs)
      std.fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j];
      for (let j = 0; j < d; j++) mean[j] /= n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          const v = X[i * d + j] - mean[j];
          std[j] += v * v;
        }
      }
      for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / Math.max(n - 1, 1)) + 1e-6;
    }
    const Xs = new Float32Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Xs[i * d + j] = (X[i * d + j] - mean[j]) / std[j];
    let wsum = 0;
    if (o.weight) for (let i = 0; i < n; i++) wsum += o.weight[i];
    const wscale = o.weight ? n / wsum : 1;

    const P = d * C + C;
    let theta = o.init && o.init.length === P ? Float64Array.from(o.init) : new Float64Array(P);
    const logit = new Float64Array(C);

    // Loss and gradient at t
    function evaluate(t, grad) {
      grad.fill(0);
      let loss = 0;
      const bo = d * C;
      for (let i = 0; i < n; i++) {
        const r = i * d;
        for (let c = 0; c < C; c++) logit[c] = t[bo + c];
        for (let j = 0; j < d; j++) {
          const v = Xs[r + j];
          const w = j * C;
          for (let c = 0; c < C; c++) logit[c] += v * t[w + c];
        }
        let mx = logit[0];
        for (let c = 1; c < C; c++) if (logit[c] > mx) mx = logit[c];
        let sum = 0;
        for (let c = 0; c < C; c++) {
          logit[c] = Math.exp(logit[c] - mx);
          sum += logit[c];
        }
        const wi = (o.weight ? o.weight[i] * wscale : 1) / n;
        loss -= wi * Math.log(logit[y[i]] / sum);
        for (let c = 0; c < C; c++) logit[c] = wi * (logit[c] / sum - (c === y[i] ? 1 : 0));
        for (let c = 0; c < C; c++) grad[bo + c] += logit[c];
        for (let j = 0; j < d; j++) {
          const v = Xs[r + j];
          const w = j * C;
          for (let c = 0; c < C; c++) grad[w + c] += v * logit[c];
        }
      }
      let pen = 0;
      for (let k = 0; k < d * C; k++) {
        pen += t[k] * t[k];
        grad[k] += o.l2 * t[k];
      }
      return loss + 0.5 * o.l2 * pen;
    }

    let grad = new Float64Array(P);
    let loss = evaluate(theta, grad);
    const S = [];
    const Y = [];
    const rho = [];
    let iter = 0;
    let done = false;

    function gmax(g) {
      let m = 0;
      for (let k = 0; k < P; k++) m = Math.max(m, Math.abs(g[k]));
      return m;
    }
    function dot(a, b) {
      let v = 0;
      for (let k = 0; k < P; k++) v += a[k] * b[k];
      return v;
    }

    function iterate() {
      if (gmax(grad) <= o.gtol) {
        done = true;
        return;
      }
      // Two-Loop Recursion: search direction -H grad
      const q = Float64Array.from(grad);
      const alpha = new Float64Array(S.length);
      for (let i = S.length - 1; i >= 0; i--) {
        alpha[i] = rho[i] * dot(S[i], q);
        for (let k = 0; k < P; k++) q[k] -= alpha[i] * Y[i][k];
      }
      if (S.length) {
        const last = S.length - 1;
        const gamma = dot(S[last], Y[last]) / dot(Y[last], Y[last]);
        for (let k = 0; k < P; k++) q[k] *= gamma;
      }
      for (let i = 0; i < S.length; i++) {
        const beta = rho[i] * dot(Y[i], q);
        for (let k = 0; k < P; k++) q[k] += (alpha[i] - beta) * S[i][k];
      }
      let slope = -dot(grad, q);
      if (slope >= 0) {  // not a descent direction: restart from steepest descent
        S.length = Y.length = rho.length = 0;
        for (let k = 0; k < P; k++) q[k] = grad[k];
        slope = -dot(grad, grad);
      }
      // Backtracking Line Search (Armijo)
      let step = iter === 0 ? Math.min(1, 1 / Math.sqrt(dot(grad, grad))) : 1;
      const next = new Float64Array(P);
      const gnext = new Float64Array(P);
      let fnext = loss;
      let ok = false;
      for (let ls = 0; ls < 30; ls++) {
        for (let k = 0; k < P; k++) next[k] = theta[k] - step * q[k];
        fnext = evaluate(next, gnext);
        if (fnext <= loss + 1e-4 * step * slope) {
          ok = true;
          break;
        }
        step *= 0.5;
      }
      if (!ok) {
        done = true;
        return;
      }
      const s = new Float64Array(P);
      const yv = new Float64Array(P);
      for (let k = 0; k < P; k++) {
        s[k] = next[k] - theta[k];
        yv[k] = gnext[k] - grad[k];
      }
      const sy = dot(s, yv);
      if (sy > 1e-12) {
        S.push(s);
        Y.push(yv);
        rho.push(1 / sy);
        if (S.length > o.history) {
          S.shift();
          Y.shift();
          rho.shift();
        }
      }
      const drop = loss - fnext;
      theta = next;
      grad = gnext;
      loss = fnext;
      iter++;
      if (drop < o.ftol * Math.max(1, Math.abs(loss)) || iter >= o.maxIter) done = true;
    }

    function step(ms) {
      const t0 = Date.now();
      while (!done && (ms == null || Date.now() - t0 < ms)) iterate();
      return { done, iter, loss };
    }

    // Weights in raw input coordinates: logits = x W + b
    function result() {
      const W = new Float64Array(d * C);
      const b = new Float64Array(C);
      for (let c = 0; c < C; c++) b[c] = theta[d * C + c];
      for (let j = 0; j < d; j++) {
        for (let c = 0; c < C; c++) {
          W[j * C + c] = theta[j * C + c] / std[j];
          b[c] -= (mean[j] / std[j]) * theta[j * C + c];
        }
      }
      return { W, b, d, C, iter, loss, theta: Float64Array.from(theta) };
    }
    return { step, result };
  }

  function fitSoftmax(X, n, d, y, C, opts) {
    const tr = makeTrainer(X, n, d, y, C, opts);
    tr.step(null);
    return tr.result();
  }

  // Class probabilities (n, C) of a fitted model
  function probabilities(model, X, n) {
    const { W, b, d, C } = model;
    const out = new Float64Array(n * C);
    for (let i = 0; i < n; i++) {
      const r = i * d;
      const o = i * C;
      for (let c = 0; c < C; c++) out[o + c] = b[c];
      for (let j = 0; j < d; j++) {
        const v = X[r + j];
        const w = j * C;
        for (let c = 0; c < C; c++) out[o + c] += v * W[w + c];
      }
      let mx = out[o];
      for (let c = 1; c < C; c++) if (out[o + c] > mx) mx = out[o + c];
      let sum = 0;
      for (let c = 0; c < C; c++) {
        out[o + c] = Math.exp(out[o + c] - mx);
        sum += out[o + c];
      }
      for (let c = 0; c < C; c++) out[o + c] /= sum;
    }
    return out;
  }

  function predict(model, X, n) {
    const { W, b, d, C } = model;
    const pred = new Uint8Array(n);
    const logit = new Float64Array(C);
    for (let i = 0; i < n; i++) {
      const r = i * d;
      for (let c = 0; c < C; c++) logit[c] = b[c];
      for (let j = 0; j < d; j++) {
        const v = X[r + j];
        const w = j * C;
        for (let c = 0; c < C; c++) logit[c] += v * W[w + c];
      }
      let best = 0;
      for (let c = 1; c < C; c++) if (logit[c] > logit[best]) best = c;
      pred[i] = best;
    }
    return pred;
  }

  // ---------------------------------------------------------------- metrics
  // Accuracy, recall of every (class, group) cell, per-class gap (group 1 - group 0), RMS gap over classes
  function recallSummary(y, z, pred, n, C) {
    const hit = new Float64Array(C * 2);
    const cnt = new Float64Array(C * 2);
    let correct = 0;
    for (let i = 0; i < n; i++) {
      const c = y[i] * 2 + z[i];
      cnt[c]++;
      if (pred[i] === y[i]) {
        hit[c]++;
        correct++;
      }
    }
    const tpr = new Float64Array(C * 2);
    const gap = new Float64Array(C);
    let sq = 0;
    for (let c = 0; c < C; c++) {
      tpr[c * 2] = cnt[c * 2] ? hit[c * 2] / cnt[c * 2] : NaN;
      tpr[c * 2 + 1] = cnt[c * 2 + 1] ? hit[c * 2 + 1] / cnt[c * 2 + 1] : NaN;
      gap[c] = tpr[c * 2 + 1] - tpr[c * 2];
      sq += gap[c] * gap[c];
    }
    return { acc: correct / n, tpr, gap, gapRms: Math.sqrt(sq / C), cnt };
  }

  return {
    makeRng, cellStats, covariance, symEig, symPower, matmul, leftSingular, biasMatrix, bmeEraser, applyEraser,
    minimalPlan, planSources, bmaCopies, makeTrainer, fitSoftmax, probabilities, predict, recallSummary,
  };
});
