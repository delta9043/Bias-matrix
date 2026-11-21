/* Demo 1: 2-D toy data (two classes, two groups).  The model, BME and BMA are computed in the page with bm-core.js. */
(function () {
  "use strict";
  const { $, el, segmented, fmt } = window.BMUI;
  const BM = window.BM;
  const SVG = "http://www.w3.org/2000/svg";

  // Toy Geometry: class centers, group shift (group 1 - group 0) inside each class, noise
  const P = { center: [[0, 0.8], [0, -0.8]], delta: [[1.9, 0.2], [1.6, -0.15]], sd: 0.6, size: 200, share: 6 / 7, l2: 1e-2, nTest: 3000 };
  const VIEW = { w: 560, h: 440, xmin: -3.4, xmax: 3.4, ymin: -2.67, ymax: 2.67 };
  const COLORS = ["#2a78d6", "#eb6834"];
  const FIT = { l2: P.l2, standardize: false };  // penalty in the plotted coordinates
  const state = { alpha: 1, method: "none", strength: 1, frac: 1, seed: 7 };  // strength: share of the bias direction removed (BME); frac: share of the minimal plan added (BMA)
  let pools = null;   // normal draws of every cell (train) and the test sample
  let view = null;    // last computed scene
  let anim = null;

  const sx = (x) => ((x - VIEW.xmin) / (VIEW.xmax - VIEW.xmin)) * VIEW.w;
  const sy = (y) => VIEW.h - ((y - VIEW.ymin) / (VIEW.ymax - VIEW.ymin)) * VIEW.h;
  const svgEl = (tag, attrs) => {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    return node;
  };

  function cellCenter(c, g) {
    return [P.center[c][0] + (g - 0.5) * P.delta[c][0], P.center[c][1] + (g - 0.5) * P.delta[c][1]];
  }

  // Fixed Draws per Seed: the first n points of a cell stay the same when alpha changes
  function makePools(seed) {
    const rng = BM.makeRng(seed);
    const draw = (count) => Float32Array.from({ length: count * 2 }, () => rng.normal());
    const train = [0, 1, 2, 3].map(() => draw(P.size));
    const test = [0, 1, 2, 3].map(() => draw(P.nTest));
    return { train, test };
  }

  function build(eps, counts) {
    const n = counts.reduce((a, b) => a + b, 0);
    const X = new Float32Array(n * 2);
    const y = new Uint8Array(n);
    const z = new Uint8Array(n);
    let i = 0;
    counts.forEach((k, cell) => {
      const c = cell >> 1;
      const g = cell & 1;
      const m = cellCenter(c, g);
      for (let j = 0; j < k; j++, i++) {
        X[i * 2] = m[0] + P.sd * eps[cell][j * 2];
        X[i * 2 + 1] = m[1] + P.sd * eps[cell][j * 2 + 1];
        y[i] = c;
        z[i] = g;
      }
    });
    return { X, y, z, n };
  }

  // Recall summary on the test sample; accuracy weighted to the cell shares of the training data
  function evaluate(model, test, counts) {
    const s = BM.recallSummary(test.y, test.z, BM.predict(model, test.X, test.n), test.n, 2);
    const total = counts.reduce((a, b) => a + b, 0);
    s.acc = counts.reduce((a, k, cell) => a + (k / total) * s.tpr[cell], 0);
    return s;
  }

  function compute() {
    const n1 = Math.round(P.size * (0.5 + state.alpha * (P.share - 0.5)));
    const counts = [P.size - n1, n1, n1, P.size - n1];  // A group 0, A group 1, B group 0, B group 1
    const train = build(pools.train, counts);
    const test = build(pools.test, [P.nTest, P.nTest, P.nTest, P.nTest]);
    const base = BM.fitSoftmax(train.X, train.n, 2, train.y, 2, FIT);
    const scene = { counts, train, base, baseSummary: evaluate(base, test, counts), added: [0, 0, 0, 0], shown: train.X, copies: null };

    if (state.method === "bme") {
      const er = BM.bmeEraser(train.X, train.n, 2, train.y, train.z, 2, { oblique: false, k: 1 });
      scene.shown = BM.applyEraser(er, train.X, train.n, 1, state.strength);
      scene.model = BM.fitSoftmax(scene.shown, train.n, 2, train.y, 2, FIT);
      scene.summary = evaluate(scene.model, { X: BM.applyEraser(er, test.X, test.n, 1, state.strength), y: test.y, z: test.z, n: test.n }, counts);
    } else if (state.method === "bma") {
      const { mu, cnt } = BM.cellStats(train.X, train.n, 2, train.y, train.z, 2);
      const plan = BM.minimalPlan(cnt, 2).plan;
      const sources = BM.planSources(train.y, train.z, train.n, 2, plan, BM.makeRng(state.seed + 1000));
      const cp = BM.bmaCopies(train.X, 2, train.y, mu, sources, state.frac);
      const X = new Float32Array((train.n + cp.m) * 2);
      X.set(train.X);
      X.set(cp.Xc, train.n * 2);
      const y = new Uint8Array(train.n + cp.m);
      y.set(train.y);
      y.set(cp.yc, train.n);
      scene.model = BM.fitSoftmax(X, train.n + cp.m, 2, y, 2, FIT);
      scene.summary = evaluate(scene.model, test, counts);
      scene.copies = cp;
      for (let i = 0; i < cp.m; i++) scene.added[cp.yc[i] * 2 + cp.zc[i]]++;
    } else {
      scene.model = base;
      scene.summary = scene.baseSummary;
    }
    return scene;
  }

  // -------------------------------------------------------------- drawing
  // Decision line of a two-class model inside the view: [x1, y1, x2, y2] in data units
  function boundary(model) {
    const w0 = model.W[0] - model.W[1];
    const w1 = model.W[2] - model.W[3];
    const b = model.b[0] - model.b[1];
    if (Math.abs(w1) >= Math.abs(w0)) return { w: [w0, w1, b], line: [VIEW.xmin, -(w0 * VIEW.xmin + b) / w1, VIEW.xmax, -(w0 * VIEW.xmax + b) / w1] };
    return { w: [w0, w1, b], line: [-(w1 * VIEW.ymin + b) / w0, VIEW.ymin, -(w1 * VIEW.ymax + b) / w0, VIEW.ymax] };
  }

  // Part of the view predicted as class B (w.x + b < 0), as SVG polygon points
  function regionB(w) {
    const corners = [[VIEW.xmin, VIEW.ymin], [VIEW.xmax, VIEW.ymin], [VIEW.xmax, VIEW.ymax], [VIEW.xmin, VIEW.ymax]];
    const f = (p) => w[0] * p[0] + w[1] * p[1] + w[2];
    const out = [];
    corners.forEach((p, i) => {
      const q = corners[(i + 1) % 4];
      const fp = f(p);
      const fq = f(q);
      if (fp < 0) out.push(p);
      if (fp < 0 !== fq < 0) {
        const t = fp / (fp - fq);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    });
    return out.map((p) => sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" ");
  }

  function mark(c, x, y, attrs) {
    const node = c === 0 ? svgEl("circle", { r: 3.3 }) : svgEl("rect", { width: 6, height: 6, rx: 1 });
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    place(node, c, x, y);
    return node;
  }

  function place(node, c, x, y) {
    if (c === 0) {
      node.setAttribute("cx", sx(x).toFixed(1));
      node.setAttribute("cy", sy(y).toFixed(1));
    } else {
      node.setAttribute("x", (sx(x) - 3).toFixed(1));
      node.setAttribute("y", (sy(y) - 3).toFixed(1));
    }
  }

  // Scene graph for one computed state; frame(t) moves it from the previous positions (t = 0) to the new ones (t = 1)
  function draw(scene, from) {
    const svg = $("#toy-svg");
    svg.replaceChildren();
    const defs = svgEl("defs");
    const clip = svgEl("clipPath", { id: "toy-clip" });
    clip.append(svgEl("rect", { x: 0, y: 0, width: VIEW.w, height: VIEW.h, rx: 11 }));
    const head = svgEl("marker", { id: "toy-arrow", viewBox: "0 0 10 10", refX: 8.5, refY: 5, markerWidth: 5.5, markerHeight: 5.5, orient: "auto-start-reverse" });
    head.append(svgEl("path", { d: "M 0 1 L 9 5 L 0 9 z", fill: "#141413" }));
    defs.append(clip, head);
    const root = svgEl("g", { "clip-path": "url(#toy-clip)" });
    svg.append(defs, root);

    const cur = boundary(scene.model);
    const region = svgEl("polygon", { points: regionB(cur.w), fill: "#f3f2ee" });
    root.append(region);

    const { train, copies } = scene;
    const dim = copies && copies.m ? 0.5 : 0.88;
    const points = [];
    const gPoints = svgEl("g");
    for (let i = 0; i < train.n; i++) {
      const node = mark(train.y[i], train.X[i * 2], train.X[i * 2 + 1], { fill: COLORS[train.z[i]], "fill-opacity": dim, stroke: "#fff", "stroke-width": 0.6 });
      gPoints.append(node);
      points.push(node);
    }
    const gCopies = svgEl("g");
    const gLinks = svgEl("g");
    const copyNodes = [];
    const links = [];
    if (copies) {
      // A Few Example Links per Cell: drawn source -> added copy
      const perCell = {};
      for (let i = 0; i < copies.m; i++) {
        const node = mark(copies.yc[i], copies.Xc[i * 2], copies.Xc[i * 2 + 1], { fill: "#fff", stroke: COLORS[copies.zc[i]], "stroke-width": 1.3 });
        gCopies.append(node);
        copyNodes.push(node);
        const cell = copies.yc[i] * 2 + copies.zc[i];
        perCell[cell] = (perCell[cell] || 0) + 1;
        if (perCell[cell] <= 6) {
          const line = svgEl("line", { stroke: "#141413", "stroke-width": 0.8, "marker-end": "url(#toy-arrow)" });
          gLinks.append(line);
          links.push({ i, line });
          points[copies.src[i]].setAttribute("fill-opacity", 1);
          points[copies.src[i]].setAttribute("stroke", "#141413");
        }
      }
    }
    root.append(gPoints, gCopies, gLinks);

    if (state.method !== "none") {
      const b0 = boundary(scene.base).line;
      root.append(svgEl("line", { x1: sx(b0[0]), y1: sy(b0[1]), x2: sx(b0[2]), y2: sy(b0[3]), stroke: "#b4b2aa", "stroke-width": 1.5, "stroke-dasharray": "5 5" }));
    }
    const line = svgEl("line", { x1: sx(cur.line[0]), y1: sy(cur.line[1]), x2: sx(cur.line[2]), y2: sy(cur.line[3]), stroke: "#141413", "stroke-width": 2, "stroke-dasharray": "7 5", "stroke-linecap": "round" });
    root.append(line);

    function frame(t) {
      const e = t < 1 ? 1 - Math.pow(1 - t, 3) : 1;
      if (from && from.length === scene.shown.length) {
        for (let i = 0; i < train.n; i++) {
          place(points[i], train.y[i], from[i * 2] + e * (scene.shown[i * 2] - from[i * 2]), from[i * 2 + 1] + e * (scene.shown[i * 2 + 1] - from[i * 2 + 1]));
        }
      } else {
        for (let i = 0; i < train.n; i++) place(points[i], train.y[i], scene.shown[i * 2], scene.shown[i * 2 + 1]);
      }
      if (copies) {
        const pos = (i) => {
          const s = copies.src[i];
          return [train.X[s * 2] + e * (copies.Xc[i * 2] - train.X[s * 2]), train.X[s * 2 + 1] + e * (copies.Xc[i * 2 + 1] - train.X[s * 2 + 1])];
        };
        copyNodes.forEach((node, i) => {
          const p = pos(i);
          place(node, copies.yc[i], p[0], p[1]);
          node.setAttribute("opacity", Math.min(1, 0.25 + e));
        });
        links.forEach(({ i, line: ln }) => {
          const s = copies.src[i];
          const p = pos(i);
          const dx = sx(p[0]) - sx(train.X[s * 2]);
          const dy = sy(p[1]) - sy(train.X[s * 2 + 1]);
          const len = Math.hypot(dx, dy) || 1;
          ln.setAttribute("x1", sx(train.X[s * 2]) + (dx / len) * 4.5);
          ln.setAttribute("y1", sy(train.X[s * 2 + 1]) + (dy / len) * 4.5);
          ln.setAttribute("x2", sx(p[0]) - (dx / len) * 5);
          ln.setAttribute("y2", sy(p[1]) - (dy / len) * 5);
          ln.setAttribute("opacity", len > 14 ? 1 : 0);
        });
      }
      line.setAttribute("opacity", from ? e : 1);
      region.setAttribute("opacity", from ? e : 1);
    }
    return frame;
  }

  // -------------------------------------------------------------- readout
  function readout(scene) {
    const s = scene.summary;
    const b = scene.baseSummary;
    const changed = state.method !== "none";
    const cellNames = ["class A", "class B"];
    const th = (text) => el("th", { scope: "col", text });

    const data = el("table", { class: "mini" }, [
      el("thead", {}, [el("tr", {}, [th(""), th("group 0"), th("group 1")])]),
      el("tbody", {}, [0, 1].map((c) => el("tr", {}, [el("td", { text: cellNames[c] })].concat([0, 1].map((g) => {
        const td = el("td", { text: String(scene.counts[c * 2 + g]) });
        if (scene.added[c * 2 + g]) td.append(el("span", { class: "added", text: " +" + scene.added[c * 2 + g] }));
        return td;
      }))))),
    ]);

    const recall = el("table", { class: "mini" }, [
      el("thead", {}, [el("tr", {}, [th(""), th("group 0"), th("group 1"), th("격차")])]),
      el("tbody", {}, [0, 1].map((c) => el("tr", {}, [el("td", { text: cellNames[c] })].concat([0, 1].map((g) => {
        const td = el("td", { text: fmt.pct(s.tpr[c * 2 + g], 0) });
        if (changed) td.append(el("span", { class: "was", text: " ← " + fmt.pct(b.tpr[c * 2 + g], 0) }));
        return td;
      })).concat([el("td", { text: (100 * Math.abs(s.gap[c])).toFixed(0) + "%p" })])))),
    ]);

    const gap = el("div", { class: "bigstat" }, [el("span", { class: "value", text: fmt.num(s.gapRms, 3) })]);
    const acc = el("div", { class: "bigstat" }, [el("span", { class: "value", text: fmt.pct(s.acc, 1) })]);
    if (changed) {
      const rel = (s.gapRms - b.gapRms) / b.gapRms;
      gap.append(el("span", { class: "delta " + (rel < 0 ? "down" : "up"), text: `(원본 대비 ${rel < 0 ? "−" : "+"}${Math.abs(100 * rel).toFixed(0)}%)` }));
      acc.append(el("span", { class: "delta", text: `(원본 대비 ${fmt.signedPct(s.acc - b.acc, 1)})` }));
    }

    $("#toy-readout").replaceChildren(
      el("div", {}, [el("h4", { text: "학습 데이터의 구성 (표본 수)" }), data]),
      el("div", {}, [el("h4", { text: "재현율 (테스트 표본)" }), recall]),
      el("div", { class: "stat-pair" }, [
        el("div", {}, [el("h4", { text: "그룹 간 재현율 격차" }), gap]),
        el("div", {}, [el("h4", { text: "정확도", title: "학습 데이터와 같은 그룹 비율의 테스트 표본에서 잰 값" }), acc]),
      ]),
    );
  }

  function update(animate) {
    const prev = view;
    view = compute();
    // Tween Only between Views of the Same Points (method switch or replay)
    const samePoints = prev && prev.train.n === view.train.n && prev.seed === state.seed && prev.alpha === state.alpha;
    const from = animate && samePoints ? prev.shown : null;
    view.seed = state.seed;
    view.alpha = state.alpha;
    const frame = draw(view, animate ? from || view.shown : null);
    readout(view);
    if (anim) cancelAnimationFrame(anim);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!animate || reduce) {
      frame(1);
      return;
    }
    const t0 = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / 900);
      frame(t);
      if (t < 1) anim = requestAnimationFrame(tick);
    };
    frame(0);
    anim = requestAnimationFrame(tick);
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("#demo-toy") || !BM) return;
    pools = makePools(state.seed);
    const alpha = $("#toy-alpha");
    const frac = $("#toy-frac");
    alpha.addEventListener("input", () => {
      state.alpha = +alpha.value;
      $("#toy-alpha-out").textContent = state.alpha.toFixed(2);
      update(false);
    });
    frac.addEventListener("input", () => {
      state.frac = +frac.value;
      $("#toy-frac-out").textContent = Math.round(100 * state.frac) + "%";
      update(false);
    });
    const strength = $("#toy-strength");
    strength.addEventListener("input", () => {
      state.strength = +strength.value;
      $("#toy-strength-out").textContent = Math.round(100 * state.strength) + "%";
      update(false);
    });
    // Second Slider beside the Bias Strength: only for the chosen method
    segmented($("#toy-method"), (value) => {
      state.method = value;
      $("#toy-strength-control").hidden = value !== "bme";
      $("#toy-frac-control").hidden = value !== "bma";
      update(true);
    });
    // Replay: the same view again, animated from the original data
    $("#toy-replay").addEventListener("click", () => {
      if (state.method === "none") return;
      const method = state.method;
      state.method = "none";
      view = compute();
      view.seed = state.seed;
      view.alpha = state.alpha;
      state.method = method;
      update(true);
    });
    $("#toy-resample").addEventListener("click", () => {
      state.seed += 1;
      pools = makePools(state.seed);
      update(false);
    });
    update(false);
  });
})();
