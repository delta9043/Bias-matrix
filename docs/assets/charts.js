/* Charts drawn in the page: SVG sized to its container (text keeps its size on every screen), lines with interval bands,
   paths and single marks, crosshair or nearest-mark tooltips, legend highlighting. */
(function () {
  "use strict";
  const { el, tooltip } = window.BMUI;
  const NS = "http://www.w3.org/2000/svg";
  const INK = "#141413";
  const INK2 = "#52514e";
  const MUTED = "#7d7b75";
  const GRID = "#eceae4";
  const AXIS = "#c3c2b7";

  function svg(tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (text != null) node.textContent = text;
    return node;
  }

  // Tick values inside [lo, hi] at a round step
  function niceTicks(lo, hi, target) {
    const span = hi - lo;
    const mag = Math.pow(10, Math.floor(Math.log10(span / target)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= target + 1) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // Tooltip body: a title and rows of (swatch, name, value)
  function tipBody(title, rows) {
    const body = el("div", { class: "tt" }, [el("div", { class: "tt-title", text: title })]);
    rows.forEach((r) => {
      const row = el("div", { class: "tt-row" });
      if (r.color) {
        const sw = el("span", { class: "tt-swatch" + (r.hollow ? " hollow" : "") });
        sw.style.setProperty("--c", r.color);
        row.append(sw);
      }
      row.append(el("span", { class: "tt-name", text: r.name }));
      if (r.value != null) row.append(el("span", { class: "tt-val", text: r.value }));
      if (r.extra) row.append(el("span", { class: "tt-extra", text: r.extra }));
      body.append(row);
    });
    return body;
  }

  /* spec: { label, height, x: {label, domain, ticks, format, log2}, y: {label, domain, ticks, format, tipFormat},
             lines: [{key, name, color, dash, hollow, points: [{x, y, lo, hi, note}]}], marks: [{key, name, color, hollow, x, y, note}],
             trends: [{key, color, line: [x1, y1, x2, y2]}], rules: [{y, label}], notes: [{key, x, y, text, anchor}], hover: "x" | "nearest", tip(hit) } */
  function chart(container, spec) {
    const lines = spec.lines || [];
    const marks = spec.marks || [];
    const groups = new Map();
    let highlighted = null;
    let width = 0;

    function domain(axis, pick) {
      if (axis.domain) return axis.domain;
      let lo = Infinity;
      let hi = -Infinity;
      const see = (v) => {
        if (v == null) return;
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      };
      lines.forEach((s) => s.points.forEach((p) => pick(p).forEach(see)));
      marks.forEach((p) => pick(p).forEach(see));
      (spec.rules || []).forEach((r) => pick({ y: r.y }).forEach(see));
      const pad = (hi - lo || 1) * (axis.pad == null ? 0.08 : axis.pad);
      return [lo - pad, hi + pad];
    }

    function group(key) {
      if (!groups.has(key)) groups.set(key, []);
      return groups.get(key);
    }

    function applyHighlight() {
      groups.forEach((nodes, key) => nodes.forEach((n) => n.setAttribute("opacity", highlighted && key !== highlighted ? 0.14 : 1)));
    }

    function render() {
      const W = Math.max(220, Math.round(container.clientWidth));
      if (W === width) return;
      width = W;
      const H = spec.height || 264;
      const m = { l: 46, r: 14, t: 28, b: 44 };
      const iw = W - m.l - m.r;
      const ih = H - m.t - m.b;
      const tx = spec.x.log2 ? Math.log2 : (v) => v;
      const xd = domain(spec.x, (p) => [p.x]);
      const yd = domain(spec.y, (p) => [p.y, p.lo, p.hi]);
      const sx = (v) => m.l + ((tx(v) - tx(xd[0])) / (tx(xd[1]) - tx(xd[0]))) * iw;
      const sy = (v) => m.t + (1 - (v - yd[0]) / (yd[1] - yd[0])) * ih;
      const fx = spec.x.format || String;
      const fy = spec.y.format || String;
      const fyTip = spec.y.tipFormat || fy;
      groups.clear();

      const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": spec.label || "" });
      const clipId = "clip-" + Math.random().toString(36).slice(2, 9);
      const defs = svg("defs");
      const clip = svg("clipPath", { id: clipId });
      clip.append(svg("rect", { x: m.l - 7, y: m.t - 7, width: iw + 14, height: ih + 14 }));
      defs.append(clip);
      root.append(defs);

      // Grid, Axes, Labels (recessive)
      const yTicks = spec.y.ticks || niceTicks(yd[0], yd[1], H < 240 ? 4 : 5);
      yTicks.forEach((v) => {
        root.append(svg("line", { x1: m.l, x2: m.l + iw, y1: sy(v), y2: sy(v), stroke: GRID, "stroke-width": 1 }));
        root.append(svg("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end", "font-size": 11.5, fill: MUTED }, fy(v)));
      });
      const xTicks = (spec.x.ticks || niceTicks(xd[0], xd[1], iw < 260 ? 4 : 6)).filter((v) => v >= xd[0] && v <= xd[1]);
      root.append(svg("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: AXIS, "stroke-width": 1 }));
      xTicks.forEach((v) => {
        root.append(svg("line", { x1: sx(v), x2: sx(v), y1: m.t + ih, y2: m.t + ih + 4, stroke: AXIS, "stroke-width": 1 }));
        root.append(svg("text", { x: sx(v), y: m.t + ih + 18, "text-anchor": "middle", "font-size": 11.5, fill: MUTED }, fx(v)));
      });
      root.append(svg("text", { x: m.l - 38, y: 13, "font-size": 12, fill: INK2 }, spec.y.label));
      root.append(svg("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: INK2 }, spec.x.label));

      const plot = svg("g", { "clip-path": `url(#${clipId})` });
      root.append(plot);
      (spec.rules || []).forEach((r) => {
        plot.append(svg("line", { x1: m.l, x2: m.l + iw, y1: sy(r.y), y2: sy(r.y), stroke: r.strong ? "#8f8d85" : AXIS, "stroke-width": 1.2, "stroke-dasharray": r.strong ? "" : "3 3" }));
        if (r.label) root.append(svg("text", { x: m.l + iw - 2, y: sy(r.y) - 5, "text-anchor": "end", "font-size": 11.5, fill: INK2 }, r.label));
      });

      // Interval Bands, Trend Lines, Lines with Markers, Single Marks
      lines.forEach((s) => {
        const band = s.points.filter((p) => p.lo != null);
        if (band.length > 1) {
          const d = band.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.hi).toFixed(1)}`).join("") +
            band.slice().reverse().map((p) => `L${sx(p.x).toFixed(1)},${sy(p.lo).toFixed(1)}`).join("") + "Z";
          const node = svg("path", { d, fill: s.color, "fill-opacity": 0.08 });
          plot.append(node);
          group(s.key).push(node);
        }
      });
      (spec.trends || []).forEach((t) => {
        const node = svg("line", { x1: sx(t.line[0]), y1: sy(t.line[1]), x2: sx(t.line[2]), y2: sy(t.line[3]), stroke: t.color, "stroke-width": 1.6 });
        plot.append(node);
        group(t.key).push(node);
      });
      const hits = [];  // every mark that the pointer can reach
      const dot = (s, p, r) => svg("circle", { cx: sx(p.x).toFixed(1), cy: sy(p.y).toFixed(1), r, fill: s.hollow ? "#fff" : s.color, stroke: s.hollow ? s.color : "#fff", "stroke-width": s.hollow ? 1.8 : 1.5 });
      lines.forEach((s) => {
        const g = svg("g");
        const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
        g.append(svg("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash ? "5 4" : "" }));
        s.points.forEach((p) => {
          g.append(dot(s, p, 4));
          hits.push({ s, p, px: sx(p.x), py: sy(p.y) });
        });
        plot.append(g);
        group(s.key).push(g);
      });
      marks.forEach((p) => {
        const node = dot(p, p, p.small ? 4 : 5);
        plot.append(node);
        group(p.key).push(node);
        hits.push({ s: p, p, px: sx(p.x), py: sy(p.y) });
      });
      // Direct Labels at Data Coordinates
      (spec.notes || []).forEach((t) => {
        const node = svg("text", { x: sx(t.x), y: sy(t.y), "text-anchor": t.anchor || "start", "font-size": 12, fill: INK2 }, t.text);
        root.append(node);
        if (t.key) group(t.key).push(node);
      });
      applyHighlight();

      // Hover Layer: crosshair on shared x values, or the nearest mark
      const inside = hits.filter((h) => h.px >= m.l - 7 && h.px <= m.l + iw + 7 && h.py >= m.t - 7 && h.py <= m.t + ih + 7);
      const guide = svg("line", { y1: m.t, y2: m.t + ih, stroke: "#8f8d85", "stroke-width": 1, visibility: "hidden" });
      const ring = svg("circle", { r: 8.5, fill: "none", stroke: INK, "stroke-width": 1.6, visibility: "hidden" });
      const overlay = svg("rect", { x: m.l - 7, y: m.t - 7, width: iw + 14, height: ih + 14, fill: "transparent", tabindex: 0, class: "chart-hit" });
      root.append(guide, ring, overlay);
      const xs = Array.from(new Set(inside.map((h) => h.p.x))).sort((a, b) => a - b);
      let current = -1;

      function showAt(index, clientX, clientY) {
        current = index;
        const box = root.getBoundingClientRect();
        if (spec.hover === "x") {
          const x = xs[index];
          const at = inside.filter((h) => h.p.x === x && h.s.points).sort((a, b) => b.p.y - a.p.y);
          guide.setAttribute("x1", sx(x));
          guide.setAttribute("x2", sx(x));
          guide.setAttribute("visibility", "visible");
          const body = tipBody(`${spec.x.tipLabel || spec.x.label} ${fx(x)}`, at.map((h) => ({
            color: h.s.color, hollow: h.s.hollow, name: h.s.name, value: fyTip(h.p.y),
            extra: h.p.lo != null ? `[${fyTip(h.p.lo)}, ${fyTip(h.p.hi)}]` : null })));
          tooltip.show(body, clientX == null ? box.left + sx(x) : clientX, clientY == null ? box.top + m.t + 12 : clientY);
        } else {
          const h = inside[index];
          ring.setAttribute("cx", h.px);
          ring.setAttribute("cy", h.py);
          ring.setAttribute("visibility", "visible");
          const rows = spec.tip ? spec.tip(h) : [{ name: spec.x.label, value: fx(h.p.x) }, { name: spec.y.label, value: fyTip(h.p.y) }];
          tooltip.show(tipBody(h.s.name, (h.p.note ? [{ name: h.p.note }] : []).concat(rows)), clientX == null ? box.left + h.px : clientX, clientY == null ? box.top + h.py : clientY);
        }
      }
      function hide() {
        current = -1;
        guide.setAttribute("visibility", "hidden");
        ring.setAttribute("visibility", "hidden");
        tooltip.hide();
      }
      function nearest(e) {
        const box = root.getBoundingClientRect();
        const px = ((e.clientX - box.left) / box.width) * W;
        const py = ((e.clientY - box.top) / box.height) * H;
        if (spec.hover === "x") {
          let best = 0;
          xs.forEach((x, i) => {
            if (Math.abs(sx(x) - px) < Math.abs(sx(xs[best]) - px)) best = i;
          });
          return xs.length ? best : -1;
        }
        let best = -1;
        let dist = 34 * 34;
        inside.forEach((h, i) => {
          const d2 = (h.px - px) ** 2 + (h.py - py) ** 2;
          if (d2 < dist) {
            dist = d2;
            best = i;
          }
        });
        return best;
      }
      const move = (e) => {
        const i = nearest(e);
        if (i < 0) hide();
        else showAt(i, e.clientX, e.clientY);
      };
      overlay.addEventListener("pointermove", move);
      overlay.addEventListener("pointerdown", move);
      overlay.addEventListener("pointerleave", (e) => {
        if (e.pointerType !== "touch") hide();
      });
      overlay.addEventListener("blur", hide);
      overlay.addEventListener("keydown", (e) => {
        const count = spec.hover === "x" ? xs.length : inside.length;
        const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!step || !count) return;
        e.preventDefault();
        showAt(((current < 0 ? (step > 0 ? -1 : 0) : current) + step + count) % count);
      });
      container.replaceChildren(root);
    }

    render();
    if ("ResizeObserver" in window) new ResizeObserver(() => render()).observe(container);
    else window.addEventListener("resize", render);
    return {
      highlight(key) {
        highlighted = key;
        applyHighlight();
      },
    };
  }

  // Legend of series; pointing at (or tapping) an entry dims the other series in the linked charts
  function legend(container, items, charts) {
    let pinned = null;
    const set = (key) => charts.forEach((c) => c.highlight(key));
    const buttons = items.map((it) => {
      const sw = svg("svg", { viewBox: "0 0 30 12", width: 30, height: 12, "aria-hidden": "true" });
      if (it.line !== false) sw.append(svg("line", { x1: 1, x2: 29, y1: 6, y2: 6, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.dash ? "5 4" : "", "stroke-linecap": "round" }));
      sw.append(svg("circle", { cx: 15, cy: 6, r: 4, fill: it.hollow ? "#fff" : it.color, stroke: it.hollow ? it.color : "#fff", "stroke-width": it.hollow ? 1.8 : 1.2 }));
      const b = el("button", { type: "button", class: "legend-item", "aria-pressed": "false" }, [sw, el("span", { text: it.name })]);
      b.addEventListener("pointerenter", (e) => {
        if (e.pointerType !== "touch") set(it.key);
      });
      b.addEventListener("pointerleave", () => set(pinned));
      b.addEventListener("focus", () => set(it.key));
      b.addEventListener("blur", () => set(pinned));
      b.addEventListener("click", () => {
        pinned = pinned === it.key ? null : it.key;
        buttons.forEach((x, i) => x.setAttribute("aria-pressed", items[i].key === pinned ? "true" : "false"));
        set(pinned);
      });
      return b;
    });
    container.replaceChildren(...buttons);
  }

  document.addEventListener("pointerdown", (e) => {
    if (!(e.target instanceof Element) || !e.target.closest(".chart-hit, .hit")) tooltip.hide();
  });

  window.BMChart = { chart, legend, svg, tipBody, niceTicks };
})();
