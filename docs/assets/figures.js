/* Figures 2-4 of the paper drawn in the page from page-data.js (series exported from the result files by demo/build_page_data.py). */
(function () {
  "use strict";
  const { $ } = window.BMUI;
  const { chart, legend } = window.BMChart;

  const f2 = (v) => v.toFixed(2);
  const f3 = (v) => v.toFixed(3);
  const signed = (v, nd) => (v > 0 ? "+" : "") + v.toFixed(nd).replace("-", "−");
  const ALPHA = { label: "클래스와 그룹 편향 강도 (α)", tipLabel: "편향 강도 α =", domain: [-0.05, 1.05], ticks: [0, 0.25, 0.5, 0.75, 1], format: String };
  const GAP = { label: "보정 GAP", format: f2, tipFormat: f3 };
  const CHANGE = { label: "정확도 변화 (%p)", format: (v) => signed(v, 1) };
  const tradeTip = (h) => [{ name: "정확도 변화", value: signed(h.p.x, 2) + "%p" }, { name: "보정 GAP", value: f3(h.p.y) }];

  // Legend entries in a fixed order from the series that the panels draw
  function entries(order, panels) {
    const found = new Map();
    panels.forEach((p) => {
      (p.lines || []).forEach((s) => found.set(s.key, Object.assign({ line: true }, s)));
      (p.marks || []).forEach((s) => {
        if (!found.has(s.key)) found.set(s.key, Object.assign({ line: false }, s));
      });
    });
    return order.filter((k) => found.has(k)).map((k) => found.get(k));
  }

  // ------------------------------------------------------------ Figures 2-4
  function fig2(F) {
    const a = chart($("#fig2a"), { label: "편향 강도에 따른 보정 GAP (Bias in Bios)", x: ALPHA, y: GAP, lines: F.a, hover: "x" });
    const b = chart($("#fig2b"), { label: "정확도 변화와 보정 GAP (Bias in Bios, 전체 데이터)", x: CHANGE, y: GAP, lines: F.b.lines, marks: F.b.marks, hover: "nearest", tip: tradeTip });
    const c = chart($("#fig2c"), { label: "제거한 방향의 수에 따른 보정 GAP", x: { label: "제거한 방향의 수 (k)", tipLabel: "k =", log2: true, domain: [0.85, 76], ticks: [1, 2, 4, 8, 16, 28, 64], format: String },
      y: GAP, lines: F.c.lines, rules: [{ y: F.c.original, label: "Original" }], hover: "x",
      notes: [{ key: "x-top", x: 1, y: F.c.lines[1].points[0].hi + 0.012, text: "data matrix X" }, { key: "bme", x: 1, y: Math.min(...F.c.lines[0].points.slice(0, 6).map((p) => p.lo)) - 0.014, text: "bias matrix B (Ours)" }] });
    legend($("#fig2-legend"), entries(["bme", "bme-uniform", "inlp", "inlp-class", "none", "bma", "leace", "reweighing", "cda"], [{ lines: F.a }, F.b]), [a, b]);
    void c;
  }

  function fig3(F) {
    const acc = F.b.flatMap((s) => s.points.map((p) => p.y));
    const lo = Math.min(...acc), hi = Math.max(...acc);
    const a = chart($("#fig3a"), { label: "편향 강도에 따른 보정 GAP (학습 데이터 교정 기법)", x: ALPHA, y: GAP, lines: F.a, hover: "x" });
    const b = chart($("#fig3b"), { label: "편향 강도에 따른 정확도", x: ALPHA, y: { label: "정확도 (%)", domain: [lo - 0.5 * (hi - lo), hi + 0.3 * (hi - lo)], format: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2) + "%" },
      lines: F.b, hover: "x" });
    const c = chart($("#fig3c"), { label: "추가한 표본의 양에 따른 보정 GAP", x: { label: "추가한 표본의 양 (원본 대비)", domain: [-0.06, 1.06], ticks: [0, 0.25, 0.5, 0.75, 1], format: String }, y: GAP,
      lines: F.c.lines, marks: F.c.marks, hover: "nearest",
      tip: (h) => [{ name: "추가량", value: h.p.x.toFixed(3) + " N" }, { name: "보정 GAP", value: f3(h.p.y), extra: h.p.lo != null ? `[${f3(h.p.lo)}, ${f3(h.p.hi)}]` : null }] });
    legend($("#fig3-legend"), entries(["none", "reweighing", "bma", "cda"], [{ lines: F.a }, F.c]), [a, b, c]);
  }

  function fig4(F) {
    const a = chart($("#fig4a"), { label: "편향 강도에 따른 보정 GAP (위해위험 상담)", x: ALPHA, y: GAP, lines: F.a, hover: "x" });
    const b = chart($("#fig4b"), { label: "정확도 변화와 보정 GAP (위해위험 상담, 전체 데이터)", x: Object.assign({ domain: F.b.xlim, ticks: [-1.2, -0.9, -0.6, -0.3, 0] }, CHANGE),
      y: Object.assign({ domain: F.b.ylim }, GAP), lines: F.b.lines, marks: F.b.marks, hover: "nearest", tip: tradeTip });
    const pct = (v) => Math.round(100 * v) + "%";
    const xmax = Math.max(...F.c.map((s) => s.trend[2]));
    const c = chart($("#fig4c"), { label: "품목별 재현율 차이와 비대면 판매 비율", x: { label: "품목의 비대면 판매 비율", domain: [0, xmax], ticks: [0, 0.1, 0.2, 0.3, 0.4], format: pct },
      y: { label: "재현율 차이 (비대면 − 일반)", format: (v) => signed(v, 1), tipFormat: (v) => signed(v, 3) },
      marks: F.c.flatMap((s) => s.points.map((p) => Object.assign({ small: true }, s, p, { points: undefined, trend: undefined }))),
      trends: F.c.map((s) => ({ key: s.key, color: s.color, line: s.trend })), rules: [{ y: 0, strong: true }], hover: "nearest",
      tip: (h) => [{ name: "비대면 판매 비율", value: (100 * h.p.x).toFixed(1) + "%" }, { name: "재현율 차이", value: signed(h.p.y, 3) }] });
    legend($("#fig4-legend"), entries(["bme", "bme-uniform", "inlp", "leace", "none", "bma", "reweighing"], [{ lines: F.a }, F.b]), [a, b, c]);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const F = window.BM_PAGE && window.BM_PAGE.figures;
    if (!F) return;
    fig2(F.fig2);
    fig3(F.fig3);
    fig4(F.fig4);
  });
})();
