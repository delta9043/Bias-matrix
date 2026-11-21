/* Figure 1 as an animation: the steps of the paper's overview figure played on its 2-D toy sample
   (page-data.js: points, the BME projection, the BMA copies, the three fitted models and their TPR gaps).
   Tabs: B (four steps that build and decompose the bias matrix), BME and BMA (a fifth step each). */
(function () {
  "use strict";
  const { $, el, segmented } = window.BMUI;
  const { svg } = window.BMChart;
  const BM = window.BM;
  const COLORS = ["#2a78d6", "#eb6834"];  // group 0, group 1
  const INK = "#141413";
  const EDGE = "#52514e";
  const W = 760;
  const H = 420;
  const sx = (x) => 248 + x * 62;
  const sy = (y) => 206 - y * 62;
  const clamp = (v) => Math.max(0, Math.min(1, v));
  const seg = (p, a, b) => clamp((p - a) / (b - a));
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const lerp = (a, b, t) => a + (b - a) * t;
  const mix = (a, b, t) => a.map((v, i) => lerp(v, b[i], t));

  // Steps: chip and title, one-line caption, play time and hold at the end (ms), position on the shared timeline
  const STEPS = {
    embed: { name: "Embed", order: 0, ms: 4200, hold: 900, text: "클래스(A, B)와 그룹(0, 1) 데이터 임베딩 변환" },
    means: { name: "Cell means", order: 1, ms: 3200, hold: 900, text: "클래스마다 두 그룹의 평균 차이 측정" },
    matrix: { name: "Bias matrix", order: 2, ms: 3200, hold: 900, text: "평균의 차이를 열로 쌓아 조건부 편향 행렬 B 생성" },
    svd: { name: "SVD", order: 3, ms: 5200, hold: 1400, text: "B를 SVD하여 그룹 차이가 가장 큰 방향과 크기(특이값) 측정" },
    bme: { name: "BME", title: "Bias-Matrix Editing (BME)", order: 4, ms: 4800, hold: 2200, text: "임베딩에서 B의 특이값 일부 제거" },
    bma: { name: "BMA", title: "Bias-Matrix Augmentation (BMA)", order: 4, ms: 6200, hold: 2600, text: "다른 그룹의 표본을 평균의 차이만큼 옮긴 데이터를 증강시켜 재학습" },
  };
  const MODES = { B: ["embed", "means", "matrix", "svd"], BME: ["embed", "means", "matrix", "svd", "bme"], BMA: ["embed", "means", "matrix", "svd", "bma"] };

  // Right Side: matrix B (two columns), its three factors after the decomposition, the singular values as bars
  const SLOT = { x: [606, 650], top: 74, bottom: 196 };
  const ORIGIN = [584, 62];                      // top-left corner of the B glyph
  const FACTOR = { scale: 0.62, shift: -50.3 };  // B glyph shrunk into the place of U
  const DIAG = [{ x: 615, y: 70, size: 18 }, { x: 643, y: 127, size: 10 }];  // entries of the middle factor
  const BAR = { x: [622, 654], base: 372, height: 104, width: 24 };

  function init(T) {
    const stage = $("#player-stage");
    const n = T.x.length;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "조건부 편향 행렬을 만들고 SVD로 분해한 뒤 BME와 BMA를 적용하는 과정의 애니메이션" });
    const defs = svg("defs");
    const head = svg("marker", { id: "pl-arrow", viewBox: "0 0 10 10", refX: 8.5, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
    head.append(svg("path", { d: "M 0 1 L 9 5 L 0 9 z", fill: INK }));
    const clip = svg("clipPath", { id: "pl-clip" });
    clip.append(svg("rect", { x: 0, y: 0, width: 520, height: H }));
    defs.append(head, clip);
    root.append(defs);
    const scatter = svg("g", { "clip-path": "url(#pl-clip)" });
    root.append(scatter);

    // Toy Sample: bias matrix, top direction and singular values from the page's own core
    const X = new Float32Array(n * 2);
    T.x.forEach((p, i) => {
      X[i * 2] = p[0];
      X[i * 2 + 1] = p[1];
    });
    const st = BM.cellStats(X, n, 2, Uint8Array.from(T.y), Uint8Array.from(T.z), 2);
    const svd = BM.leftSingular(BM.biasMatrix(st.mu, st.cnt, 2, 2, n), 2, 2);
    const u1 = [svd.U[0], svd.U[svd.rank]];
    const center = [0, 1].map((j) => T.x.reduce((a, p) => a + p[j], 0) / n);
    const barH = [BAR.height, Math.max(6, (BAR.height * svd.s[1]) / svd.s[0])];

    const shape = (c, r) => (c === 0 ? svg("circle", { r }) : svg("rect", { width: 2 * r - 0.6, height: 2 * r - 0.6, rx: 1 }));
    const place = (node, c, x, y, r) => {
      if (c === 0) {
        node.setAttribute("cx", x.toFixed(1));
        node.setAttribute("cy", y.toFixed(1));
      } else {
        node.setAttribute("x", (x - r + 0.3).toFixed(1));
        node.setAttribute("y", (y - r + 0.3).toFixed(1));
      }
    };
    const px = (p) => [sx(p[0]), sy(p[1])];

    // Scatter Layers (bottom to top): top direction, copies, points, example links, decision boundary
    const dir = svg("line", { stroke: COLORS[0], "stroke-width": 2.2, "stroke-linecap": "round" });
    scatter.append(dir);
    const cp = T.copies;
    const copies = cp.x.map((p, i) => {
      const node = shape(cp.y[i], 3.4);
      node.setAttribute("fill", "#fff");
      node.setAttribute("stroke", COLORS[cp.z[i]]);
      node.setAttribute("stroke-width", 1.2);
      scatter.append(node);
      return node;
    });
    const points = T.x.map((p, i) => {
      const node = shape(T.y[i], 3.5);
      node.setAttribute("fill", COLORS[T.z[i]]);
      scatter.append(node);
      return node;
    });
    const linked = cp.linked.map((i) => {
      const ring = shape(T.y[cp.src[i]], 5.2);
      ring.setAttribute("fill", "none");
      ring.setAttribute("stroke", INK);
      ring.setAttribute("stroke-width", 1.4);
      const p = px(T.x[cp.src[i]]);
      place(ring, T.y[cp.src[i]], p[0], p[1], 5.2);
      const line = svg("line", { stroke: INK, "stroke-width": 1, "marker-end": "url(#pl-arrow)" });
      scatter.append(ring, line);
      return { i, ring, line };
    });
    const boundary = svg("line", { stroke: INK, "stroke-width": 2.2, "stroke-dasharray": "8 6", "stroke-linecap": "round" });
    scatter.append(boundary);
    const means = T.means.map((p, cell) => {
      const node = shape(cell >> 1, 6.5);
      node.setAttribute("fill", COLORS[cell & 1]);
      node.setAttribute("stroke", INK);
      node.setAttribute("stroke-width", 1.6);
      const q = px(p);
      place(node, cell >> 1, q[0], q[1], 6.5);
      root.append(node);
      return node;
    });
    const arrows = [0, 1].map(() => {
      const node = svg("line", { stroke: INK, "stroke-width": 2.2, "marker-end": "url(#pl-arrow)" });
      root.append(node);
      return node;
    });

    // Matrix Glyphs: B, and the factors of B = U S V^T (B itself shrinks into the place of U)
    const bracket = (x0, x1, y0, y1) => `M${x0 + 12} ${y0} H${x0} V${y1} H${x0 + 12} M${x1 - 12} ${y0} H${x1} V${y1} H${x1 - 12}`;
    const gB = svg("g");
    gB.append(svg("path", { d: bracket(584, 672, 62, 208), fill: "none", stroke: EDGE, "stroke-width": 2.4 }));
    const columns = SLOT.x.map((x) => svg("rect", { x: x - 9, y: SLOT.top, width: 18, height: SLOT.bottom - SLOT.top, rx: 3, fill: "#e9e8e2", stroke: EDGE, "stroke-width": 1.2 }));
    columns.forEach((c) => gB.append(c));
    const gS = svg("g");
    gS.append(svg("path", { d: bracket(608.7, 663.3, 62, 152.5), fill: "none", stroke: EDGE, "stroke-width": 1.6 }));
    const diag = DIAG.map((q, j) => svg("rect", { x: q.x, y: q.y, width: q.size, height: q.size, rx: 2.5, fill: j === 0 ? COLORS[0] : "#a3a199" }));
    diag.forEach((q) => gS.append(q));
    const gV = svg("g");
    gV.append(svg("path", { d: bracket(683.7, 738.3, 62, 152.5), fill: "none", stroke: EDGE, "stroke-width": 1.6 }));
    [74, 129].forEach((y) => gV.append(svg("rect", { x: 692, y, width: 38, height: 11, rx: 2.5, fill: "#e9e8e2", stroke: EDGE, "stroke-width": 0.9 })));
    const bars = BAR.x.map((_, j) => svg("rect", { rx: 3, fill: j === 0 ? COLORS[0] : "#a3a199" }));
    const axis = svg("line", { x1: 598, x2: 678, y1: BAR.base, y2: BAR.base, stroke: "#c3c2b7", "stroke-width": 1.4 });
    root.append(gV, gS, gB, axis, ...bars);

    // Text Labels over the Drawing (HTML: fixed size at every stage width)
    const label = (text, x, y) => {
      const node = el("span", { class: "player-label", text });
      node.style.left = (100 * x) / W + "%";
      node.style.top = (100 * y) / H + "%";
      return node;
    };
    const flip = Math.sign(u1[0] || 1);
    const end = [center[0] + 2.75 * u1[0] * flip, center[1] + 2.75 * u1[1] * flip];
    const labels = {
      a: label("class A", sx(-2.75), sy(2.45)), b: label("class B", sx(-2.75), sy(-2.55)), B: label("B", 628, 40), usv: label("B = U Σ Vᵀ", 636, 40),
      s1: label("s₁", BAR.x[0], BAR.base + 16), s2: label("s₂", BAR.x[1], BAR.base + 16), u: label("u₁", sx(end[0]) + 4, sy(end[1]) - 14),
    };
    stage.replaceChildren(root, ...Object.values(labels));

    // Decision Lines of the Three Models as Pixel Segments
    const segment = (m) => {
      const f = (x) => -(m.w[0] * x + m.w[2]) / m.w[1];
      return [sx(-3.6), sy(f(-3.6)), sx(3.6), sy(f(3.6))];
    };
    const lines = { original: segment(T.original), bme: segment(T.bme), bma: segment(T.bma) };
    const stagger = T.x.map((_, i) => (i * 0.6180339887) % 1);
    const show = (node, v) => node.setAttribute("opacity", v.toFixed(3));
    const gapBox = $("#player-gap");
    const gapValue = el("b");
    gapBox.replaceChildren(gapValue, el("span", { text: "TPR gap" }));

    function draw(key, p) {
      const at = STEPS[key].order;
      const isBme = key === "bme";
      const isBma = key === "bma";

      // Points: staggered appearance, projection onto the kept axis (BME), dimmed under the copies (BMA)
      const proj = isBme ? ease(seg(p, 0.1, 0.56)) : 0;
      const copyT = isBma ? ease(seg(p, 0.3, 0.66)) : 0;
      const dim = isBma ? 1 - 0.55 * seg(p, 0.12, 0.28) : 1;
      for (let i = 0; i < n; i++) {
        const a = at === 0 ? clamp(seg(p, 0, 0.5) * 1.6 - stagger[i] * 0.6) : 1;
        place(points[i], T.y[i], sx(lerp(T.x[i][0], T.edited[i][0], proj)), sy(lerp(T.x[i][1], T.edited[i][1], proj)), 3.5);
        show(points[i], 0.86 * a * dim);
      }

      // Decision Boundary and TPR Gap of the Model Fit to the Shown Training Set
      let line = lines.original;
      let gap = T.original.gap;
      const grow = at === 0 ? ease(seg(p, 0.56, 0.9)) : 1;
      if (isBme || isBma) {
        const t = isBme ? ease(seg(p, 0.5, 0.86)) : ease(seg(p, 0.68, 0.92));
        line = mix(lines.original, isBme ? lines.bme : lines.bma, t);
        gap = lerp(T.original.gap, (isBme ? T.bme : T.bma).gap, t);
      }
      const mx = (line[0] + line[2]) / 2;
      const my = (line[1] + line[3]) / 2;
      boundary.setAttribute("x1", lerp(mx, line[0], grow));
      boundary.setAttribute("y1", lerp(my, line[1], grow));
      boundary.setAttribute("x2", lerp(mx, line[2], grow));
      boundary.setAttribute("y2", lerp(my, line[3], grow));
      show(boundary, grow > 0 ? 1 : 0);
      const text = gap.toFixed(2);
      if (gapValue.textContent !== text) gapValue.textContent = text;
      gapBox.style.visibility = at === 0 && p < 0.86 ? "hidden" : "visible";

      // Cell Means and the Difference Arrows, Which Become the Columns of B
      const meanOp = at === 1 ? ease(seg(p, 0, 0.3)) : at === 2 ? 1 - 0.6 * seg(p, 0.05, 0.4) : at === 3 ? 0.4 * (1 - seg(p, 0, 0.3)) : 0;
      means.forEach((node) => show(node, meanOp));
      arrows.forEach((node, c) => {
        const from = px(T.means[c * 2]);
        const to = px(T.means[c * 2 + 1]);
        let a = from;
        let b = to;
        let op = 0;
        if (at === 1) {
          b = mix(from, to, ease(seg(p, 0.3, 0.75)));
          op = seg(p, 0.3, 0.38);
        } else if (at === 2) {
          const t = ease(seg(p, 0.08, 0.5));
          a = mix(from, [SLOT.x[c], SLOT.bottom], t);
          b = mix(to, [SLOT.x[c], SLOT.top + 8], t);
          op = 1 - seg(p, 0.52, 0.72);
        }
        node.setAttribute("x1", a[0]);
        node.setAttribute("y1", a[1]);
        node.setAttribute("x2", b[0]);
        node.setAttribute("y2", b[1]);
        show(node, op);
      });

      // Decomposition: B splits into three factors, the diagonal entries leave the middle one and become the bars
      const side = isBma ? 1 - 0.65 * seg(p, 0.04, 0.18) : 1;
      const split = at < 3 ? 0 : at === 3 ? ease(seg(p, 0.06, 0.36)) : 1;
      const out = at < 3 ? 0 : at === 3 ? ease(seg(p, 0.46, 0.76)) : 1;
      const made = at < 2 ? 0.28 : at === 2 ? lerp(0.28, 1, seg(p, 0, 0.15)) : 1;
      const scale = lerp(1, FACTOR.scale, split);
      gB.setAttribute("transform", `translate(${ORIGIN[0] + FACTOR.shift * split} ${ORIGIN[1]}) scale(${scale}) translate(${-ORIGIN[0]} ${-ORIGIN[1]})`);
      show(gB, made * side);
      columns.forEach((node) => show(node, at < 2 ? 0 : at === 2 ? seg(p, 0.52, 0.75) : 1));
      gS.setAttribute("transform", `translate(${lerp(-34, 0, split)} 0)`);
      gV.setAttribute("transform", `translate(${lerp(-84, 0, split)} 0)`);
      show(gS, split * side);
      show(gV, split * side);
      show(diag[0], 1 - 0.85 * proj);
      labels.B.style.opacity = made * (1 - split) * side;
      labels.usv.style.opacity = split * side;
      bars.forEach((node, j) => {
        const h = lerp(DIAG[j].size, barH[j] * (j === 0 ? 1 - proj : 1), out);
        const w = lerp(DIAG[j].size, BAR.width, out);
        const bottom = lerp(DIAG[j].y + DIAG[j].size, BAR.base, out);
        node.setAttribute("x", lerp(DIAG[j].x, BAR.x[j] - BAR.width / 2, out));
        node.setAttribute("y", bottom - Math.max(0.01, h));
        node.setAttribute("width", w);
        node.setAttribute("height", Math.max(0.01, h));
        show(node, (out > 0 ? 1 : 0) * side);
      });
      show(axis, out * side);
      labels.s1.style.opacity = labels.s2.style.opacity = seg(out, 0.8, 1) * side;

      // Top Direction u1 through the Data Mean
      const reach = at === 3 ? ease(seg(p, 0.72, 0.96)) : isBme ? 1 : 0;
      const dirOp = at === 3 ? (reach > 0 ? 1 : 0) : isBme ? 1 - seg(p, 0.84, 1) : 0;
      dir.setAttribute("x1", sx(center[0] - 2.75 * u1[0] * reach));
      dir.setAttribute("y1", sy(center[1] - 2.75 * u1[1] * reach));
      dir.setAttribute("x2", sx(center[0] + 2.75 * u1[0] * reach));
      dir.setAttribute("y2", sy(center[1] + 2.75 * u1[1] * reach));
      show(dir, 0.9 * dirOp);
      labels.u.style.opacity = reach >= 1 ? dirOp : 0;

      // BMA: drawn sources (ringed), then copies shifted by the group mean difference
      const copyOp = isBma ? seg(p, 0.28, 0.38) : 0;
      copies.forEach((node, i) => {
        const from = T.x[cp.src[i]];
        place(node, cp.y[i], sx(lerp(from[0], cp.x[i][0], copyT)), sy(lerp(from[1], cp.x[i][1], copyT)), 3.4);
        show(node, 0.8 * copyOp);
      });
      linked.forEach(({ i, ring, line: ln }) => {
        const from = px(T.x[cp.src[i]]);
        const to = [sx(lerp(T.x[cp.src[i]][0], cp.x[i][0], copyT)), sy(lerp(T.x[cp.src[i]][1], cp.x[i][1], copyT))];
        const len = Math.hypot(to[0] - from[0], to[1] - from[1]);
        show(ring, isBma ? seg(p, 0.1, 0.22) : 0);
        ln.setAttribute("x1", from[0]);
        ln.setAttribute("y1", from[1]);
        ln.setAttribute("x2", lerp(from[0], to[0], len > 12 ? 1 - 6 / len : 0));
        ln.setAttribute("y2", lerp(from[1], to[1], len > 12 ? 1 - 6 / len : 0));
        show(ln, len > 12 ? copyOp : 0);
      });
      labels.a.style.opacity = labels.b.style.opacity = at === 0 ? seg(p, 0.3, 0.5) : 1;
    }
    return draw;
  }

  document.addEventListener("DOMContentLoaded", () => {
    const T = window.BM_PAGE && window.BM_PAGE.figures && window.BM_PAGE.figures.fig1;
    if (!T || !$("#method-player") || !BM) return;
    const draw = init(T);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let steps = MODES.B;
    let si = 0;
    let at = 0;        // ms inside the step
    let playing = !reduce;
    let speed = 1;
    let visible = false;
    let started = reduce;
    let last = 0;
    let frame = null;
    let chips = [];

    function caption() {
      const step = STEPS[steps[si]];
      $("#player-title").textContent = `Step ${si + 1}. ${step.title || step.name}`;
      $("#player-text").textContent = step.text;
      chips.forEach((li, j) => {
        if (j === si) li.firstChild.setAttribute("aria-current", "step");
        else li.firstChild.removeAttribute("aria-current");
      });
    }
    function paint() {
      const step = STEPS[steps[si]];
      draw(steps[si], clamp(at / step.ms));
      const span = (k) => STEPS[k].ms + STEPS[k].hold;
      const before = steps.slice(0, si).reduce((a, k) => a + span(k), 0);
      $("#player-progress").style.width = (100 * (before + Math.min(at, span(steps[si])))) / steps.reduce((a, k) => a + span(k), 0) + "%";
    }
    function go(j, offset) {
      si = j;
      at = offset;
      caption();
      paint();
    }
    // Step Chips of the Chosen Tab: four shared steps, plus the fifth for BME or BMA
    function setMode(mode) {
      steps = MODES[mode];
      chips = steps.map((k, j) => {
        const b = el("button", { type: "button", text: `${j + 1}. ${STEPS[k].name}` });
        b.addEventListener("click", () => {
          go(j, reduce ? STEPS[k].ms : 0);
          schedule();
        });
        return el("li", {}, [b]);
      });
      $("#player-steps").replaceChildren(...chips);
      go(0, started && !reduce ? 0 : STEPS[steps[0]].ms);
      schedule();
    }
    function tick(now) {
      frame = null;
      const dt = Math.min(64, now - last) * speed;
      last = now;
      if (playing && visible) {
        at += dt;
        if (at >= STEPS[steps[si]].ms + STEPS[steps[si]].hold) go((si + 1) % steps.length, 0);
        else paint();
        schedule();
      }
    }
    function schedule() {
      if (frame === null && playing && visible) {
        last = performance.now();
        frame = requestAnimationFrame(tick);
      }
    }
    // Controls: play, pause, double speed, replay
    const buttons = { play: $("#player-play"), pause: $("#player-pause"), fast: $("#player-fast") };
    function state(next) {
      if ("playing" in next) playing = next.playing;
      if ("speed" in next) speed = next.speed;
      buttons.play.setAttribute("aria-pressed", playing && speed === 1 ? "true" : "false");
      buttons.pause.setAttribute("aria-pressed", playing ? "false" : "true");
      buttons.fast.setAttribute("aria-pressed", playing && speed === 2 ? "true" : "false");
      schedule();
    }
    buttons.play.addEventListener("click", () => state({ playing: true, speed: 1 }));
    buttons.pause.addEventListener("click", () => state({ playing: false }));
    buttons.fast.addEventListener("click", () => state({ playing: true, speed: 2 }));
    $("#player-restart").addEventListener("click", () => {
      go(0, 0);
      state({ playing: true });
    });
    segmented($("#player-mode"), setMode);

    // Plays Only While on Screen; before the first play the stage shows the finished first step
    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      if (visible && !started) {
        started = true;
        go(0, 0);
      }
      schedule();
    }, { threshold: 0.25 }).observe($("#player-stage"));
    setMode("B");
    state({});
  });
})();
