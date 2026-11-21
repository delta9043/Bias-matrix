"""Numbers, tables and figure data of the project page from the results: docs/assets/page-data.js.

The page draws Figures 2-4 and the method animation (Figure 1) itself from the series exported here
(same selections as experiments/figures.py).

    python -m experiments.tables          # results/tables.json
    python -m demo.build_page_data
"""
import json

import numpy as np

from bias_matrix import bme_eraser, cell_stats, conditional_bias_matrix, count_table, mean_shift, minimal_plan
from experiments.common import PALETTE, ROOT, load_json
from experiments.figures import SERIES, by_name, family

OUT = ROOT / "docs" / "assets" / "page-data.js"


def style(key, name=None, color=None):
    """Identity of one series on the page: the label, color and line style of the paper figures."""
    label, col, _, ls, hollow = SERIES[key]
    return {"key": key, "name": name or label, "color": color or col, "dash": ls == "--", "hollow": hollow}


def r4(v):
    return round(float(v), 4)


# ------------------------------------------------------------------ tables
def table_rows(table):
    """Table rows for the page: block titles spelled out, LaTeX-free cells, corrected gap with its interval as numbers (bar in the gap column)."""
    titles = table.get("blocks", {})
    return [{"label": r["label"], "cells": [c.replace("$", "") for c in r["cells"]], "block": titles.get(r["block"]) or r["block"], "ours": r["ours"],
             "gap": [r4(r["values"]["gap_rms_bc"]), *(r4(v) for v in r["values"]["gap_rms_ci"])]} for r in table["rows"]]


# ----------------------------------------------------------------- figures
def dose_lines(res, methods, key="gap_rms_bc"):
    """One metric vs. bias strength alpha: mean over the subsample draws; corrected gap with the mean 95% interval."""
    out = []
    for name in methods:
        sel = [[r for r in res["rows"] if r["method"] == name and r["alpha"] == al] for al in res["alphas"]]
        v = np.array([[r[key] for r in rs] for rs in sel]).mean(1) * (100 if key == "acc" else 1)
        pts = [{"x": al, "y": r4(val)} for al, val in zip(res["alphas"], v)]
        if key == "gap_rms_bc":
            ci = np.array([[r["gap_rms_ci"] for r in rs] for rs in sel]).mean(1)
            pts = [p | {"lo": r4(lo), "hi": r4(hi)} for p, (lo, hi) in zip(pts, ci)]
        out.append(style(name) | {"points": pts})
    return out


def tradeoff(trade, paths, points, from_k=1):
    """Accuracy change vs. corrected gap on the full data: BME paths over the number of removed directions,
    further paths (key, rows, reference, note) and single methods (key, row, reference)."""
    base = next(r for r in trade["rows"] if r["method"] == "none")

    def point(r, ref, note=None):
        return {"x": round(100 * (r["acc"] - ref["acc"]), 3), "y": r4(r["gap_rms_bc"])} | ({"note": note} if note else {})

    def bme_path(method):
        rows = sorted((r for r in trade["rows"] if r["method"] == method and r["k"] >= from_k), key=lambda r: r["k"])
        return rows, base, "방향 {}개 제거", "k"

    lines = []
    for key, (rows, ref, note, param) in [("bme", bme_path("bme")), ("bme-uniform", bme_path("bme-uniform")), *paths]:
        start = {"x": 0.0, "y": r4(ref["gap_rms_bc"]), "note": "원본 모델"}
        lines.append(style(key) | {"points": [start] + [point(r, ref, note.format(r[param])) for r in rows]})
    marks = [style(key) | point(r, ref) for key, r, ref in [*points, ("none", base, base)]]
    return {"lines": lines, "marks": marks}


def interval_line(sty, xs, rows, notes=None):
    pts = [{"x": float(x), "y": r4(r["gap_rms_bc"]), "lo": r4(r["gap_rms_ci"][0]), "hi": r4(r["gap_rms_ci"][1])} for x, r in zip(xs, rows)]
    if notes:
        pts = [p | {"note": n} for p, n in zip(pts, notes)]
    return sty | {"points": pts}


def overview():
    """Figure 1 (method animation): the toy example of experiments.figures.fig_overview (same generator and seed)."""
    rng = np.random.default_rng(5)
    center = np.array([[0.0, 1.0], [0.0, -1.0]])
    delta = np.array([[1.9, 0.3], [1.5, -0.25]])

    def sample(counts):
        y, z = np.repeat([0, 0, 1, 1], counts), np.repeat([0, 1, 0, 1], counts)
        return center[y] + (z - 0.5)[:, None] * delta[y] + rng.normal(0, 0.55, (len(y), 2)), y, z

    def fit(X, y, l2=1e-2):
        A, t, w = np.c_[X, np.ones(len(X))], (y == 0).astype(float), np.zeros(3)
        for _ in range(30):
            p = 1 / (1 + np.exp(-A @ w))
            grad = A.T @ (p - t) / len(t) + l2 * np.r_[w[:2], 0]
            hess = (A * (p * (1 - p))[:, None]).T @ A / len(t) + l2 * np.diag([1, 1, 0])
            w -= np.linalg.solve(hess, grad)
        return w

    def gap(w, X, y, z):
        hit = (np.c_[X, np.ones(len(X))] @ w > 0) == (y == 0)
        return np.sqrt(np.mean([(hit[(y == c) & (z == 1)].mean() - hit[(y == c) & (z == 0)].mean()) ** 2 for c in (0, 1)]))

    X, y, z = sample([20, 120, 120, 20])
    Xt, yt, zt = sample([5000] * 4)
    eraser = bme_eraser(conditional_bias_matrix(X, y, z, 2, 2), X.mean(0), k=1)
    plan, _ = minimal_plan(count_table(y, z, 2, 2))
    src = np.concatenate([rng.choice(np.flatnonzero((y == c) & (z == 1 - g)), plan[c, g]) for c in (0, 1) for g in (0, 1)])
    means, _ = cell_stats(X, y, z, 2, 2)
    shifted = mean_shift(X, y, z, src, 1 - z[src], means)
    pick = np.random.default_rng(7)
    linked = np.concatenate([pick.choice(np.flatnonzero(y[src] == c), 5, replace=False) for c in (0, 1)])

    def model(train_X, train_y, test_X):
        w = fit(train_X, train_y)
        return {"w": [r4(v) for v in w], "gap": round(float(gap(w, test_X, yt, zt)), 2)}

    xy = lambda A: np.round(A, 3).tolist()  # noqa: E731
    return {
        "y": y.tolist(), "z": z.tolist(), "x": xy(X), "edited": xy(eraser(X)), "means": xy(means.reshape(4, 2)),
        "copies": {"x": xy(shifted), "src": src.tolist(), "y": y[src].tolist(), "z": (1 - z[src]).tolist(), "linked": linked.tolist()},
        "original": model(X, y, Xt), "bme": model(eraser(X), y, eraser(Xt)), "bma": model(np.r_[X, shifted], np.r_[y, y[src]], Xt),
    }


def figures():
    removal, aug, hazard = load_json("removal.json"), load_json("augmentation.json"), load_json("hazard.json")
    dose = {d: load_json(f"dose_{d}.json") for d in ("bios", "hazard")}
    trade = {d: load_json(f"tradeoff_{d}.json") for d in ("bios", "hazard")}
    names, data, hz = by_name(removal["rows"]), by_name(aug["rows"]), by_name(hazard["rows"])
    rounds = lambda fam: (family(removal["rows"], fam)[1], names["none"], "반복 {}회", "param")  # noqa: E731

    # Figure 2 (Bias in Bios): dose, trade-off, number of directions
    fig2 = {
        "a": dose_lines(dose["bios"], ("none", "leace", "inlp", "bme")),
        "b": tradeoff(trade["bios"], [("inlp", rounds("inlp")), ("inlp-class", rounds("inlp-class"))],
                      [("leace", names["leace"], names["none"]), ("reweighing", data["reweighing"], data["none"]),
                       ("bma", data["bma"], data["none"]), ("cda", data["cda-all"], data["none"])]),
        "c": {"lines": [interval_line(style("bme", "bias matrix B (Ours)"), *family(removal["rows"], "bme-orth")),
                        interval_line(style("reweighing", "data matrix X", PALETTE[3]) | {"key": "x-top"}, *family(removal["rows"], "X-top"))],
              "original": r4(names["none"]["gap_rms_bc"])},
    }

    # Figure 3 (Bias in Bios, training-data methods): dose (gap, accuracy), amount of added samples
    added = lambda *keys: [0.0] + [data[k]["added_over_n"] for k in keys]  # noqa: E731
    fig3 = {
        "a": dose_lines(dose["bios"], ("none", "reweighing", "bma")),
        "b": dose_lines(dose["bios"], ("none", "reweighing", "bma"), "acc"),
        "c": {"lines": [interval_line(style("bma"), added("bma", "bma-all"), [data["none"], data["bma"], data["bma-all"]],
                                      ["원본 모델", "최소 계획의 칸에만 추가", "모든 표본의 사본 추가"]),
                        interval_line(style("cda"), added("cda-partial", "cda-all"), [data["none"], data["cda-partial"], data["cda-all"]],
                                      ["원본 모델", "CDA (partial)", "CDA (all)"])],
              "marks": [style("reweighing") | {"x": 0.0, "y": r4(data["reweighing"]["gap_rms_bc"])}]},
    }

    # Figure 4 (hazard consultations): dose, trade-off (zoomed as in the paper), per-item recall difference
    inlp = next(r for r in trade["hazard"]["rows"] if r["method"] == "inlp")
    base = next(r for r in trade["hazard"]["rows"] if r["method"] == "none")
    share = np.array(hazard["share"])
    items = []
    for name in ("none", "bme"):
        gap = np.array(hz[name]["gap"])
        slope, icpt = np.polyfit(share, gap, 1)
        items.append(style(name) | {"points": [{"x": r4(s), "y": r4(g)} for s, g in zip(share, gap)],
                                    "trend": [0.0, r4(icpt), r4(share.max() * 1.08), r4(icpt + slope * share.max() * 1.08)]})
    fig4 = {
        "a": dose_lines(dose["hazard"], ("none", "leace", "inlp", "bme", "bma")),
        "b": tradeoff(trade["hazard"], [], [("leace", hz["leace"], hz["none"]), ("inlp", inlp, base), ("reweighing", hz["reweighing"], hz["none"]),
                                            ("bma", hz["bma"], hz["none"])], from_k=4) | {"xlim": [-1.3, 0.08], "ylim": [0.150, 0.215]},
        "c": items,
    }
    return {"fig1": overview(), "fig2": fig2, "fig3": fig3, "fig4": fig4}


def main():
    tables = load_json("tables.json")
    numbers = {key: item["value"] for key, item in tables["numbers"].items()}
    # Original Model of Table 2 (accuracy, corrected GAP)
    original = next(r for r in tables["table2"]["rows"] if r["block"] == "original")
    numbers |= {"bios.none.acc": original["cells"][1], "bios.none.gap": original["cells"][2].split()[0]}

    page = {"numbers": numbers,
            "table2": {"columns": tables["table2"]["columns"], "rows": table_rows(tables["table2"])},
            "table3": {"columns": tables["table3"]["columns"], "rows": table_rows(tables["table3"])},
            "figures": figures()}
    OUT.write_text("window.BM_PAGE = " + json.dumps(page, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    toy = page["figures"]["fig1"]
    print(f"{OUT.name}: {len(numbers)} numbers, {len(page['table2']['rows'])} + {len(page['table3']['rows'])} table rows, 4 figures | "
          f"{OUT.stat().st_size / 1e3:.0f} kB | Figure 1 TPR gaps: original {toy['original']['gap']}, BME {toy['bme']['gap']}, BMA {toy['bma']['gap']}")


if __name__ == "__main__":
    main()
