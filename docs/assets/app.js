/* Page behaviour: numbers and tables from page-data.js, math rendering, navigation state, shared UI helpers for the demos. */
(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    });
    (children || []).forEach((c) => node.append(c));
    return node;
  }

  // ------------------------------------------------------------ shared UI
  const tooltip = {
    node: null,
    show(text, x, y) {
      const t = this.node || (this.node = $("#tooltip"));
      if (typeof text === "string") t.textContent = text;
      else t.replaceChildren(text);
      t.hidden = false;
      const w = t.offsetWidth;
      const h = t.offsetHeight;
      t.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x + 14)) + "px";
      t.style.top = (y + h + 24 > window.innerHeight ? y - h - 12 : y + 16) + "px";
    },
    hide() {
      if (this.node) this.node.hidden = true;
    },
    // Hover and keyboard focus on one element; text() is read when shown
    attach(node, text) {
      node.addEventListener("pointermove", (e) => this.show(text(), e.clientX, e.clientY));
      node.addEventListener("pointerleave", () => this.hide());
      node.addEventListener("focus", () => {
        const r = node.getBoundingClientRect();
        this.show(text(), r.left + r.width / 2, r.top + r.height / 2);
      });
      node.addEventListener("blur", () => this.hide());
    },
  };

  // Radio-group behaviour for a .segmented element; returns a setter
  function segmented(root, onChange) {
    const buttons = $$("button", root);
    function set(value, silent) {
      buttons.forEach((b) => {
        const on = b.dataset.value === value;
        b.setAttribute("aria-checked", on ? "true" : "false");
        b.tabIndex = on ? 0 : -1;
      });
      if (!silent) onChange(value);
    }
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => set(b.dataset.value));
      b.addEventListener("keydown", (e) => {
        const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
        if (!step) return;
        e.preventDefault();
        const next = buttons[(i + step + buttons.length) % buttons.length];
        next.focus();
        set(next.dataset.value);
      });
    });
    set(buttons.find((b) => b.getAttribute("aria-checked") === "true").dataset.value, true);
    return set;
  }

  const fmt = {
    pct: (v, nd) => (100 * v).toFixed(nd == null ? 1 : nd) + "%",
    num: (v, nd) => v.toFixed(nd == null ? 3 : nd),
    int: (v) => Math.round(v).toLocaleString("en-US"),
    signedPct: (v, nd) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(100 * v).toFixed(nd == null ? 1 : nd) + "%p",
  };

  window.BMUI = { $, $$, el, tooltip, segmented, fmt };

  // ------------------------------------------------------ numbers and tables
  function fillNumbers(page) {
    $$("[data-v]").forEach((node) => {
      const v = page.numbers[node.dataset.v];
      if (v === undefined) {
        console.warn("page-data.js has no value for", node.dataset.v);
        return;
      }
      if (node.textContent.trim() !== String(v)) console.warn("fallback text differs from page-data.js:", node.dataset.v, node.textContent, v);
      node.textContent = v;
    });
  }

  function renderTable(target, table) {
    const head = el("tr", {}, table.columns.map((c) => el("th", { scope: "col", text: c })));
    const body = el("tbody");
    const gapCol = table.columns.findIndex((c) => c.startsWith("보정 GAP")) - 1;  // cell index of the corrected gap
    const scale = 1.08 * Math.max(...table.rows.map((r) => (r.gap ? r.gap[2] : 0)));
    let block = null;
    table.rows.forEach((row, i) => {
      const grouped = table.groupRows && row.block !== block;
      if (grouped) body.append(el("tr", { class: "group-row" }, [el("td", { colspan: table.columns.length, text: row.block })]));
      const classes = [row.ours ? "ours" : "", !table.groupRows && i > 0 && row.block !== block ? "block-start" : ""].join(" ").trim();
      const cells = row.cells.map((c, j) => {
        const text = c.replace(/^-(?=\d)/, "−").replace(/^--$/, "–");
        if (j !== gapCol || !row.gap) return el("td", { text });
        // Corrected Gap as a Bar with Its 95% Interval beside the Number
        const bar = el("span", { class: "gapbar", "aria-hidden": "true" }, [el("i"), el("b")]);
        bar.firstChild.style.width = (100 * row.gap[0]) / scale + "%";
        bar.lastChild.style.left = (100 * row.gap[1]) / scale + "%";
        bar.lastChild.style.width = (100 * (row.gap[2] - row.gap[1])) / scale + "%";
        return el("td", {}, [el("div", { class: "gapcell" }, [bar, el("span", { text })])]);
      });
      body.append(el("tr", { class: classes || null }, [el("td", { text: row.label })].concat(cells)));
      block = row.block;
    });
    target.replaceChildren(el("table", { class: "results" }, [el("thead", {}, [head]), body]));
  }

  // ----------------------------------------------------------- navigation
  function trackSections() {
    const links = $$(".nav-links a");
    const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const visible = new Set();
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id)));
      const current = [...byId.keys()].find((id) => visible.has(id));
      links.forEach((a) => a.removeAttribute("aria-current"));
      if (current) byId.get(current).setAttribute("aria-current", "true");
    }, { rootMargin: "-45% 0px -50% 0px" });
    byId.forEach((_, id) => {
      const s = document.getElementById(id);
      if (s) obs.observe(s);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const page = window.BM_PAGE;
    if (page) {
      fillNumbers(page);
      renderTable($("#table2"), page.table2);
      renderTable($("#table3"), Object.assign({ groupRows: true }, page.table3));
    }
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        throwOnError: false,
      });
    }
    trackSections();
    const copy = $("#copy-code");
    if (copy) {
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText($("#quickstart").innerText);
          copy.textContent = "복사됨";
        } catch (e) {
          copy.textContent = "복사 실패";
        }
        setTimeout(() => (copy.textContent = "복사"), 1600);
      });
    }
  });
})();
