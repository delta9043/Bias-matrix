/* Demo 2: real embeddings (seven medical occupations of Bias in Bios, PCA-reduced).  Training, BME and BMA run in the page. */
(function () {
  "use strict";
  const { $, el, segmented, fmt, tooltip } = window.BMUI;
  const BM = window.BM;
  const SVG = "http://www.w3.org/2000/svg";
  const KO = { surgeon: "외과의사", chiropractor: "카이로프랙터", dentist: "치과의사", physician: "의사", psychologist: "심리학자", nurse: "간호사", dietitian: "영양사" };
  const COLORS = ["#2a78d6", "#eb6834"];  // male, female
  const L2 = 1e-3;
  // Examples Shown (of the fourteen in the data): one that every model gets right, five whose prediction depends on the gendered words
  const SHOWN = ["surgeon-1", "surgeon-2", "nurse-2", "ambiguous-1", "ambiguous-3", "ambiguous-4"];

  const state = { method: "none", k: 7, retrain: true, frac: 1 };
  const cache = new Map();
  let data = null;      // decoded train / test / examples
  let base = null;      // original model and its results
  let shared = null;    // eraser, cell statistics and plan sources (computed once)
  let job = 0;
  let started = false;
  let lastChart = null; // result shown in the recall chart (redrawn when the width changes)

  const svgEl = (tag, attrs, text) => {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (text != null) node.textContent = text;
    return node;
  };
  const status = (text) => ($("#bios-status").textContent = text);
  const name = (c) => KO[data.classes[c]] || data.classes[c];

  // ------------------------------------------------------------------ data
  function bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function decode(raw) {
    const d = raw.meta.dim;
    const part = (p) => {
      const q = new Int8Array(bytes(p.x).buffer);
      const n = q.length / d;
      const X = new Float32Array(n * d);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = q[i * d + j] * raw.scale[j];
      return { X, n, y: bytes(p.y), z: bytes(p.z) };
    };
    // Example Rows: female version then male version of every text
    const m = raw.examples.length;
    const EX = new Float32Array(m * 2 * d);
    raw.examples.forEach((e, i) => {
      EX.set(e.x.f, (2 * i) * d);
      EX.set(e.x.m, (2 * i + 1) * d);
    });
    return { d, C: raw.meta.classes.length, classes: raw.meta.classes, train: part(raw.train), test: part(raw.test), examples: raw.examples, EX, nEx: m * 2 };
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("load failed"));
      document.head.append(s);
    });
  }

  // -------------------------------------------------------------- training
  // Softmax regression in time slices, so the page stays responsive
  function train(X, n, y, token, label) {
    const { d, C } = data;
    const trainer = BM.makeTrainer(X, n, d, y, C, { l2: L2, init: base ? base.model.theta : null });
    return new Promise((resolve) => {
      const tick = () => {
        if (token !== job) return resolve(null);
        const r = trainer.step(16);
        status(`${label} · 반복 ${r.iter}회`);
        if (r.done) resolve(trainer.result());
        else setTimeout(tick, 0);
      };
      tick();
    });
  }

  function summarize(model, Xte, EX, extra) {
    const { test, C, nEx } = data;
    return Object.assign({
      model,
      summary: BM.recallSummary(test.y, test.z, BM.predict(model, Xte, test.n), test.n, C),
      probs: BM.probabilities(model, EX, nEx),
      added: new Int32Array(C * 2),
      nAdded: 0,
    }, extra || {});
  }

  async function solve(token) {
    const { train: tr, test: te, d, C, EX, nEx } = data;
    const key = state.method === "none" || (state.method === "bme" && state.k === 0) || (state.method === "bma" && state.frac === 0)
      ? "none" : state.method === "bme" ? `bme:${state.k}:${state.retrain}` : `bma:${state.frac}`;
    if (key === "none") return base;
    if (cache.has(key)) return cache.get(key);
    let result;
    if (state.method === "bme") {
      if (!shared.eraser) shared.eraser = BM.bmeEraser(tr.X, tr.n, d, tr.y, tr.z, C, { oblique: true });
      const er = shared.eraser;
      const model = state.retrain ? await train(BM.applyEraser(er, tr.X, tr.n, state.k), tr.n, tr.y, token, "편집한 임베딩으로 학습 중") : base.model;
      if (!model) return null;
      result = summarize(model, BM.applyEraser(er, te.X, te.n, state.k), BM.applyEraser(er, EX, nEx, state.k));
    } else {
      const cp = BM.bmaCopies(tr.X, d, tr.y, shared.mu, shared.sources, state.frac);
      const X = new Float32Array((tr.n + cp.m) * d);
      X.set(tr.X);
      X.set(cp.Xc, tr.n * d);
      const y = new Uint8Array(tr.n + cp.m);
      y.set(tr.y);
      y.set(cp.yc, tr.n);
      const model = await train(X, tr.n + cp.m, y, token, "사본을 더한 데이터로 학습 중");
      if (!model) return null;
      const added = new Int32Array(C * 2);
      for (let i = 0; i < cp.m; i++) added[cp.yc[i] * 2 + cp.zc[i]]++;
      result = summarize(model, te.X, EX, { added, nAdded: cp.m });
    }
    cache.set(key, result);
    return result;
  }

  // ------------------------------------------------------------- rendering
  function renderKpis(cur) {
    const changed = cur !== base;
    const s = cur.summary;
    const b = base.summary;
    const n = data.train.n;
    const kpi = (label, value, sub) => el("div", { class: "kpi" }, [el("p", { class: "label", text: label }), el("p", { class: "value", text: value }), el("p", { class: "sub", text: sub })]);
    const rel = (s.gapRms - b.gapRms) / b.gapRms;
    const third = state.method === "bma" && cur.nAdded
      ? kpi("학습 데이터", `${fmt.int(n)} + ${fmt.int(cur.nAdded)}건`, `원본의 ${(100 * cur.nAdded / n).toFixed(1)}% 추가`)
      : state.method === "bme" && changed
        ? kpi("학습 데이터", `${fmt.int(n)}건`, `방향 ${state.k}개 제거 · ${state.retrain ? "재학습" : "모델 고정"}`)
        : kpi("학습 데이터", `${fmt.int(n)}건`, "편집과 증강 없음");
    $("#bios-kpis").replaceChildren(
      kpi("그룹 간 재현율 격차", fmt.num(s.gapRms, 3), changed ? `원본 ${fmt.num(b.gapRms, 3)} 대비 ${rel < 0 ? "−" : "+"}${Math.abs(100 * rel).toFixed(0)}%` : "원본 모델"),
      kpi("정확도", fmt.pct(s.acc, 1), changed ? `원본 ${fmt.pct(b.acc, 1)} 대비 ${fmt.signedPct(s.acc - b.acc, 1)}` : "원본 모델"),
      third,
    );
  }

  // Dumbbell per occupation: recall of male and female biographies; thin marks above = original model
  function renderChart(cur) {
    const { C, train: tr } = data;
    const changed = cur !== base;
    // Drawn at the Container Width: text keeps its size on narrow screens
    const W = Math.max(300, Math.round($("#bios-chart").clientWidth)), rowH = 36, top = 6, left = W < 400 ? 112 : 132, right = W - 52, H = top + C * rowH + 28;
    lastChart = cur;
    const x = (v) => left + v * (right - left);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "직업별, 성별 재현율" });
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      svg.append(svgEl("line", { x1: x(v), y1: top, x2: x(v), y2: top + C * rowH, stroke: "#e1e0d9", "stroke-width": 1 }));
      svg.append(svgEl("text", { x: x(v), y: H - 8, "text-anchor": "middle", "font-size": 12, fill: "#7d7b75" }, Math.round(100 * v) + "%"));
    });
    svg.append(svgEl("text", { x: W - 2, y: H - 8, "text-anchor": "end", "font-size": 12, fill: "#7d7b75" }, "격차"));
    for (let c = 0; c < C; c++) {
      const yc = top + c * rowH + rowH / 2;
      const share = shared.cnt[c * 2 + 1] / (shared.cnt[c * 2] + shared.cnt[c * 2 + 1]);
      svg.append(svgEl("text", { x: 0, y: yc - 3, "font-size": 13, "font-weight": 650, fill: "#141413" }, name(c)));
      svg.append(svgEl("text", { x: 0, y: yc + 11, "font-size": 11, fill: "#7d7b75" }, `학습 표본의 여성 ${(100 * share).toFixed(0)}%`));
      const t = cur.summary.tpr;
      const t0 = base.summary.tpr;
      if (changed) {
        svg.append(svgEl("line", { x1: x(t0[c * 2]), y1: yc - 7, x2: x(t0[c * 2 + 1]), y2: yc - 7, stroke: "#c9c7bf", "stroke-width": 1.5 }));
        [0, 1].forEach((g) => svg.append(svgEl("circle", { cx: x(t0[c * 2 + g]), cy: yc - 7, r: 3.4, fill: "#fff", stroke: COLORS[g], "stroke-width": 1.4, opacity: 0.75 })));
      }
      const y1 = changed ? yc + 5 : yc;
      svg.append(svgEl("line", { x1: x(t[c * 2]), y1, x2: x(t[c * 2 + 1]), y2: y1, stroke: "#141413", "stroke-width": 2, "stroke-linecap": "round" }));
      [0, 1].forEach((g) => svg.append(svgEl("circle", { cx: x(t[c * 2 + g]), cy: y1, r: 6, fill: COLORS[g], stroke: "#fff", "stroke-width": 2 })));
      svg.append(svgEl("text", { x: W - 2, y: yc + 4, "text-anchor": "end", "font-size": 13, "font-weight": 650, fill: "#141413" }, (100 * Math.abs(cur.summary.gap[c])).toFixed(0) + "%p"));
      if (c > 0) svg.append(svgEl("line", { x1: 0, y1: top + c * rowH, x2: W, y2: top + c * rowH, stroke: "#eeede8", "stroke-width": 1 }));
      const hit = svgEl("rect", { class: "hit", x: 0, y: top + c * rowH, width: W, height: rowH, fill: "transparent", tabindex: 0 });
      const cnt = cur.summary.cnt;
      tooltip.attach(hit, () => [
        `${name(c)} (${data.classes[c]})`,
        `남성 ${fmt.pct(t[c * 2])} · 테스트 ${fmt.int(cnt[c * 2])}건`,
        `여성 ${fmt.pct(t[c * 2 + 1])} · 테스트 ${fmt.int(cnt[c * 2 + 1])}건`,
        `격차 ${(100 * Math.abs(cur.summary.gap[c])).toFixed(1)}%p`,
      ].concat(changed ? [`원본 모델: 남성 ${fmt.pct(t0[c * 2])} · 여성 ${fmt.pct(t0[c * 2 + 1])}`] : []).join("\n"));
      svg.append(hit);
    }
    $("#bios-chart").replaceChildren(svg);
  }

  function renderComposition(cur) {
    const { C } = data;
    const th = (text) => el("th", { scope: "col", text });
    const rows = [];
    for (let c = 0; c < C; c++) {
      const n0 = shared.cnt[c * 2];
      const n1 = shared.cnt[c * 2 + 1];
      const a0 = cur.added[c * 2];
      const a1 = cur.added[c * 2 + 1];
      const cell = (n, a) => {
        const td = el("td", { text: fmt.int(n) });
        if (a) td.append(el("span", { class: "added", text: " +" + fmt.int(a) }));
        return td;
      };
      const before = n1 / (n0 + n1);
      const after = (n1 + a1) / (n0 + n1 + a0 + a1);
      const share = el("td", { text: (100 * after).toFixed(0) + "%" });
      if (a0 || a1) share.append(el("span", { class: "was", text: " ← " + (100 * before).toFixed(0) + "%" }));
      rows.push(el("tr", {}, [el("td", { text: name(c) }), cell(n0, a0), cell(n1, a1), share]));
    }
    $("#bios-table").replaceChildren(el("table", { class: "mini" }, [
      el("thead", {}, [el("tr", {}, [th("직업"), th("남성"), th("여성"), th("여성 비율")])]),
      el("tbody", {}, rows),
    ]));
  }

  // Text of the female version with the words that the gender swap changes marked
  function markedText(e) {
    const f = e.text.f.split(" ");
    const m = e.text.m.split(" ");
    const p = el("p", { class: "text", title: "누르면 전체 문장을 펼친다" });
    p.addEventListener("click", () => p.classList.toggle("open"));
    f.forEach((word, i) => {
      if (i) p.append(" ");
      if (m[i] !== word) p.append(el("mark", { title: "남성형: " + m[i], text: word }));
      else p.append(word);
    });
    return p;
  }

  function renderExamples(cur) {
    const { C, examples } = data;
    const changed = cur !== base;
    const top = (probs, row) => {
      let best = 0;
      for (let c = 1; c < C; c++) if (probs[row * C + c] > probs[row * C + best]) best = c;
      return { c: best, p: probs[row * C + best] };
    };
    const cards = examples.map((e, i) => ({ e, i })).filter(({ e }) => SHOWN.includes(e.id)).map(({ e, i }, shownAt) => {
      const line = (label, probs) => {
        const f = top(probs, 2 * i);
        const m = top(probs, 2 * i + 1);
        const pred = (t) => {
          const td = el("td", {}, [el("span", { class: e.label !== null && t.c !== e.label ? "wrong" : "", text: name(t.c) }), el("span", { class: "p", text: fmt.pct(t.p, 0) })]);
          return td;
        };
        const same = f.c === m.c;
        return el("tr", {}, [el("td", { class: "model", text: label }), pred(f), pred(m), el("td", {}, [el("span", { class: "badge " + (same ? "same" : "diff"), text: same ? "일치" : "불일치" })])]);
      };
      const rows = [line("원본", base.probs)].concat(changed ? [line("현재", cur.probs)] : []);
      const th = (text) => el("th", { scope: "col", text });
      return el("article", { class: "example" }, [
        el("p", { class: "meta" }, [el("span", { class: "tag", text: "예시 " + (shownAt + 1) }), e.label === null ? "정답 없음 (일부러 모호하게 쓴 약력)" : "정답: " + name(e.label)]),
        markedText(e),
        el("table", { class: "pred" }, [
          el("thead", {}, [el("tr", {}, [th(""), th("여성형 문장"), th("남성형 문장"), th("예측 일치 여부")])]),
          el("tbody", {}, rows),
        ]),
      ]);
    });
    $("#bios-examples").replaceChildren(...cards);
  }

  function render(cur) {
    renderKpis(cur);
    renderChart(cur);
    renderComposition(cur);
    renderExamples(cur);
  }

  async function run() {
    const token = ++job;
    const cur = await solve(token);
    if (!cur || token !== job) return;
    render(cur);
    status("");
  }

  // ------------------------------------------------------------------ setup
  function showControls() {
    $("#bios-k-control").hidden = $("#bios-retrain-control").hidden = state.method !== "bme";
    $("#bios-frac-control").hidden = state.method !== "bma";
  }

  async function start() {
    if (started) return;
    started = true;
    const button = $("#bios-load");
    button.disabled = true;
    button.textContent = "데이터를 불러오는 중…";
    try {
      if (!window.BM_DEMO_DATA) await loadScript("assets/demo-data.js");
      data = decode(window.BM_DEMO_DATA);
    } catch (e) {
      started = false;
      button.disabled = false;
      button.textContent = "불러오지 못했다. 다시 시도";
      return;
    }
    const { train: tr, test: te, d, C, EX } = data;
    $("#bios-k").max = C;
    $("#bios-k").value = state.k = C;
    $("#bios-k-out").textContent = C;
    $("#bios-start").hidden = true;
    $("#bios-main").hidden = false;

    // Original Model, Cell Statistics, Minimal Plan and Its Sources
    const { mu, cnt } = BM.cellStats(tr.X, tr.n, d, tr.y, tr.z, C);
    const plan = BM.minimalPlan(cnt, C).plan;
    shared = { mu, cnt, sources: BM.planSources(tr.y, tr.z, tr.n, C, plan, BM.makeRng(1)) };
    const token = ++job;
    const model = await train(tr.X, tr.n, tr.y, token, "원본 모델 학습 중");
    if (!model) return;
    base = summarize(model, te.X, EX);
    render(base);
    status("");
    if ("ResizeObserver" in window) {
      let seen = $("#bios-chart").clientWidth;
      new ResizeObserver(() => {
        const w = $("#bios-chart").clientWidth;
        if (w !== seen && lastChart) renderChart(lastChart);
        seen = w;
      }).observe($("#bios-chart"));
    }

    segmented($("#bios-method"), (value) => {
      state.method = value;
      showControls();
      run();
    });
    let timer = null;
    const later = () => {
      clearTimeout(timer);
      timer = setTimeout(run, 180);
    };
    $("#bios-k").addEventListener("input", (e) => {
      state.k = +e.target.value;
      $("#bios-k-out").textContent = state.k;
      later();
    });
    $("#bios-frac").addEventListener("input", (e) => {
      state.frac = +e.target.value;
      $("#bios-frac-out").textContent = Math.round(100 * state.frac) + "%";
      later();
    });
    $("#bios-retrain").addEventListener("change", (e) => {
      state.retrain = e.target.checked;
      run();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("#demo-bios") || !BM) return;
    $("#bios-load").addEventListener("click", start);
    // Start When the Demo Scrolls into View
    if ("IntersectionObserver" in window) {
      const obs = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          obs.disconnect();
          start();
        }
      }, { rootMargin: "200px 0px" });
      obs.observe($("#demo-bios"));
    }
  });
})();
