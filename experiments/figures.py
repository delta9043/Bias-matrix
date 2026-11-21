"""Figures 1-4 of the paper, drawn from the result JSON files (sized for the 5.5 in text width)."""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from bias_matrix import bme_eraser, cell_stats, conditional_bias_matrix, count_table, mean_shift, minimal_plan

from .common import AXIS, FIG_DIR, INK, INK2, MUTED, PALETTE, RESULTS_DIR, SURFACE, setup_style

WIDTH = 5.5  # text width of the NeurIPS style (in); figures are included at 1:1
PAPER_RC = {
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 5.8, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "lines.linewidth": 1.2, "lines.markersize": 3.6, "lines.markeredgewidth": 0.5, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.labelpad": 2, "axes.titlepad": 3, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.2, "savefig.dpi": 300, "savefig.pad_inches": 0.02,
}
NOTE = 5.6  # font size of in-plot annotations

# Fixed Identity of the Series in the Result Figures: key -> (label, color, marker, line style, hollow marker)
# One Color per Method, Circle Markers Only; a variant keeps the hue of its method (dashed line, open circles)
SERIES = {
    "none": ("Original", INK2, "o", "-", False),
    "leace": ("LEACE", PALETTE[1], "o", "-", False),
    "inlp": ("INLP", PALETTE[2], "o", "-", False),
    "inlp-class": ("INLP (직업 보정)", PALETTE[2], "o", "--", True),
    "bme": ("BME (Ours)", PALETTE[0], "o", "-", False),
    "bme-uniform": ("BME (Ours, 클래스 균등)", PALETTE[0], "o", "--", True),
    "bma": ("BMA (Ours)", PALETTE[6], "o", "-", False),
    "reweighing": ("Reweighing", PALETTE[3], "o", "-", False),
    "cda": ("CDA", PALETTE[4], "o", "-", False),
}
KOREAN_RC = {"font.family": ["Segoe UI", "Malgun Gothic", "DejaVu Sans"]}  # Hangul axis labels by glyph fallback (family list, not the generic name)
BAND = 0.07  # opacity of the 95% interval bands
WIDE_MIDDLE = {"width_ratios": [0.78, 1.44, 0.78]}  # panel (b) wide: the trade-off paths lie close together


def family(rows, name):
    """Rows of one method family sorted by parameter: (params, rows)."""
    sel = sorted((r for r in rows if r["family"] == name), key=lambda r: r["param"])
    return np.array([r["param"] for r in sel]), sel


def by_name(rows):
    return {r["method"]: r for r in rows}


def hline(ax, val, text, va="bottom", x=0.99):
    ax.axhline(val, color=AXIS, lw=0.7)
    ax.annotate(text, (x, val), xycoords=("axes fraction", "data"), ha="right" if x > 0.5 else "left", va=va, fontsize=NOTE, color=INK2)


def finish(fig, name, **pad):
    fig.tight_layout(**({"pad": 0.3, "w_pad": 0.9, "h_pad": 0.8} | pad))
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


# ------------------------------------------------------------------- Figure 1
def fig_overview(spec):
    """Figure 1: method overview. Top band: steps that build the bias matrix B; bottom band: the two uses of B.

    Scatters: a 2-D toy example (two classes, two groups) processed with the library routines; dashed line = logistic
    model fit to the training set of that panel, number = its RMS TPR gap on a large toy test sample.  Step 4 bars: measured spectrum.
    """
    rng = np.random.default_rng(5)
    center = np.array([[0.0, 1.0], [0.0, -1.0]])  # class A (top), B (bottom)
    delta = np.array([[1.9, 0.3], [1.5, -0.25]])  # group 1 - group 0 shift inside each class

    def sample(counts):
        """Toy embeddings for the cell counts (A group 0, A group 1, B group 0, B group 1)."""
        y, z = np.repeat([0, 0, 1, 1], counts), np.repeat([0, 1, 0, 1], counts)
        return center[y] + (z - 0.5)[:, None] * delta[y] + rng.normal(0, 0.55, (len(y), 2)), y, z

    def fit(X, y, l2=1e-2):
        """Ridge-penalized logistic regression by Newton steps: (w1, w2, b), positive side = class A."""
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

    X, y, z = sample([20, 120, 120, 20])  # A mostly group 1, B mostly group 0
    Xt, yt, zt = sample([5000] * 4)
    mu = X.mean(0)
    eraser = bme_eraser(conditional_bias_matrix(X, y, z, 2, 2), mu, k=1)
    plan, _ = minimal_plan(count_table(y, z, 2, 2))  # additions per (class, group) cell
    src = np.concatenate([rng.choice(np.flatnonzero((y == c) & (z == 1 - g)), plan[c, g]) for c in (0, 1) for g in (0, 1)])
    means, _ = cell_stats(X, y, z, 2, 2)
    shifted = mean_shift(X, y, z, src, 1 - z[src], means)
    xs = np.array([-3.4, 3.4])

    def toy(a, shown, train, test, copies=None, yc=None, zc=None, links=(), alpha=0.8):
        """One toy scatter: originals (opacity `alpha`), added copies (hollow), a few (source, copy, class, source group, copy group)
        links at full strength (copy None: only the drawn source, ringed; else a thin arrow to the copy), the model fit to `train`
        and its TPR gap on `test`.  Returns the fitted weights (w1, w2, b)."""
        marks = ("o", "s")  # class A, class B
        if copies is not None:
            for c in (0, 1):
                for g in (0, 1):
                    sel = (yc == c) & (zc == g)
                    a.plot(copies[sel, 0], copies[sel, 1], marks[c], ms=2.2, mfc="none", mec=PALETTE[g], mew=0.4, alpha=0.45)
        for c in (0, 1):
            for g in (0, 1):
                sel = (y == c) & (z == g)
                a.plot(shown[sel, 0], shown[sel, 1], marks[c], color=PALETTE[g], ms=2.0, alpha=alpha, mew=0.2 if alpha > 0.5 else 0)
        for p0, p1, c, g0, g1 in links:  # how a copy is made: drawn source sample -> added copy
            if p1 is None:
                a.plot(*p0, marks[c], color=PALETTE[g0], ms=2.9, mec=INK, mew=0.45)
                continue
            a.annotate("", p1, p0, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.3, "shrinkA": 1.3, "shrinkB": 1.3, "mutation_scale": 3})
            a.plot(*p0, marks[c], color=PALETTE[g0], ms=2.3, mew=0)
            a.plot(*p1, marks[c], ms=2.5, mfc=SURFACE, mec=PALETTE[g1], mew=0.6)
        w = fit(*train)
        a.plot(xs, -(w[0] * xs + w[2]) / w[1], "--", color=INK, lw=0.8)
        a.annotate(f"TPR gap: {gap(w, test, yt, zt):.2f}", (0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom", fontsize=NOTE, color=INK2)
        a.set(xlim=xs, ylim=(-3.5, 2.7), xticks=[], yticks=[])  # free strip at the bottom for the labels
        a.grid(False)
        for side in a.spines.values():
            side.set_visible(False)
        return w

    # Canvas in Units of 0.055 in (100 wide, y from Y0 to H): same scale in both directions
    W, H, Y0 = 100, 71, 6.2
    fig = plt.figure(figsize=(WIDTH, WIDTH * (H - Y0) / W))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set(xlim=(0, W), ylim=(Y0, H))
    bg.axis("off")

    def inset(x0, y0, x1, y1):
        return fig.add_axes([x0 / W, (y0 - Y0) / (H - Y0), (x1 - x0) / W, (y1 - y0) / (H - Y0)])

    def box(x0, y0, x1, y1, solid=False):
        bg.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5", fc="#f3f2ee" if solid else "none",
                                    ec=INK2 if solid else MUTED, lw=0.6, ls="-" if solid else (0, (3, 2))))

    def arrow(p0, p1):
        bg.annotate("", p1, p0, arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.9, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7})

    def text(x, y_, s, size=6.0, weight="normal", ha="center", color=INK):
        bg.text(x, y_, s, fontsize=size, fontweight=weight, ha=ha, va="center", color=color)

    # Top Band: steps 1-4 build the bias matrix
    text(1, 69.3, "Building the bias matrix $B$", 7.5, "bold", ha="left")
    steps = [(1, 19.5, "Step 1: Embed"), (23.5, 48, "Step 2: Cell means"), (52, 75, "Step 3: Bias matrix"), (79, 99, "Step 4: SVD and rank")]
    for x0, x1, name in steps:
        box(x0, 39.5, x1, 61)
        text((x0 + x1) / 2, 63.4, name, 6.6, "bold")
    for (_, x1, _), (x0, _, _) in zip(steps, steps[1:]):
        arrow((x1 + 0.7, 50.2), (x0 - 0.7, 50.2))

    text(10.25, 58.3, "396k biographies")
    arrow((10.25, 56.7), (10.25, 54.5))
    box(3.2, 48.5, 17.3, 54.2, solid=True)
    text(10.25, 52.5, "frozen LLM")
    text(10.25, 50.1, "(Qwen3-0.6B)", NOTE, color=INK2)
    arrow((10.25, 48.2), (10.25, 46.0))
    text(10.25, 44.3, "embedding $x_i\\in\\mathbb{R}^{d}$")
    text(10.25, 41.6, "class: $y_i$,  group: $z_i$", NOTE, color=INK2)

    a = inset(24.3, 44.4, 47.2, 60.5)
    w = toy(a, X, (X, y), Xt)
    # Direct Label Along the Dashed Line (rotated to its slope on the page)
    line = lambda x: -(w[0] * x + w[2]) / w[1]  # noqa: E731
    p = a.transData.transform([[2.0, line(2.0)], [3.0, line(3.0)]])
    a.annotate("boundary", (3.3, line(3.3)), xytext=(0, -1.5), textcoords="offset points", ha="right", va="top", fontsize=NOTE - 0.6,
               color=INK2, rotation=np.degrees(np.arctan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0])), rotation_mode="anchor")
    for c, (name, pos, va) in enumerate([("class A", 0.98, "top"), ("class B", 0.02, "bottom")]):  # arrows: columns of B before scaling
        a.annotate("", means[c, 1], means[c, 0], arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.6, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 5})
        a.annotate(name, (0.02, pos), xycoords="axes fraction", ha="left", va=va, fontsize=NOTE, color=INK2)
    text(35.75, 41.9, "$\\delta_y=\\mu_{1,y}-\\mu_{0,y}$ for every class $y$", NOTE, color=INK2)

    for j in range(8):  # matrix glyph: one column per (class, group) pair
        bg.add_patch(Rectangle((57.6 + 1.5 * j, 47.2), 1.1, 11.6, fc="#e6e5df", ec=INK2, lw=0.4))
    for x_, d in ((56.6, 0.8), (70.3, -0.8)):  # brackets
        bg.plot([x_ + d, x_, x_, x_ + d], [46.6, 46.6, 59.4, 59.4], color=INK2, lw=0.7)
    text(54.9, 53.0, "$d$", NOTE, color=INK2)
    text(63.5, 44.9, "$B=[\\,b_{z,y}\\,]\\in\\mathbb{R}^{d\\times CG}$")
    text(63.5, 41.9, "$y=1,\\dots,C;\\ \\ z=0,\\dots,G-1$", NOTE, color=INK2)

    sv, thr, k = np.array(spec["sv"]), np.array(spec["thr"]), spec["k"]
    a = inset(81.5, 47.3, 97.6, 59.6)  # measured spectrum of B with the sequential permutation threshold
    idx = np.arange(1, len(sv) + 1)
    a.bar(idx, sv, width=0.7, color=[PALETTE[0] if j <= k else MUTED for j in idx])
    a.plot(idx, thr, "--", color=INK, lw=0.7)
    a.set(yscale="log", xticks=[], yticks=[])
    a.minorticks_off()
    a.grid(False)
    for side in a.spines.values():
        side.set_visible(False)
    a.annotate("singular values $s_j$\ntop-$k$ keep", (0.98, 0.98), xycoords="axes fraction", ha="right", va="top", fontsize=NOTE, color=INK2)
    text(89, 44.9, "$B=U\\,\\mathrm{diag}(s)\\,V^{T}$")
    text(89, 41.9, "$k$: sequential permutation test", NOTE, color=INK2)

    # Bottom Band: the two uses of B on the same toy data
    bg.plot([1, 99], [36.3, 36.3], ls=(0, (4, 3)), color=MUTED, lw=0.7)
    text(1, 33.9, "Debiasing with $B$", 7.5, "bold", ha="left")
    text(16, 30.4, "Route 1: Bias-Matrix Editing (BME)", 6.6, "bold")
    text(67.5, 30.4, "Route 2: Bias-Matrix Augmentation (BMA)", 6.6, "bold")
    for x0, x1 in ((1, 31), (36, 66), (69, 99)):
        box(x0, 7.0, x1, 28.4)
    toy(inset(2, 13.4, 30, 27.8), eraser(X), (eraser(X), y), eraser(Xt))
    text(16, 11.3, "BME: $x'=x-U_kU_k^{T}(x-\\bar{x})$")
    text(16, 8.8, "zeroing the top $k$ singular values of $B$", NOTE, color=INK2)

    # Route 2 Panels, the Two Steps of BMA: 2a draw samples of another group in the class, 2b shift them by the group mean difference
    pick = np.random.default_rng(7)  # display choice of the example sources (the model of 2b uses every copy)
    linked = np.concatenate([pick.choice(np.flatnonzero(y[src] == c), 5, replace=False) for c in (0, 1)])
    toy(inset(37, 13.4, 65, 27.8), X, (X, y), Xt, links=[(X[src[i]], None, y[src[i]], z[src[i]], None) for i in linked], alpha=0.35)
    text(51, 11.3, "2a: draw samples of the other group")
    arrow((66.5, 17.7), (68.5, 17.7))
    toy(inset(70, 13.4, 98, 27.8), X, (np.r_[X, shifted], np.r_[y, y[src]]), Xt, copies=shifted, yc=y[src], zc=1 - z[src],
        links=[(X[src[i]], shifted[i], y[src[i]], z[src[i]], 1 - z[src[i]]) for i in linked], alpha=0.35)
    text(84, 11.3, "2b: mean-shift copies: $\\tilde{x}=x\\pm\\delta_y$")
    text(84, 8.8, "add data to make class and group independent", 5.2, color=INK2)

    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[0], ms=3, label="group 0"),
               plt.Line2D([], [], marker="o", ls="", color=PALETTE[1], ms=3, label="group 1"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, mew=0.6, ms=3, label="added copy"),
               plt.Line2D([], [], ls="-", color=INK, lw=0.6, marker=">", ms=2.2, label="group shift $\\delta_y$"),
               plt.Line2D([], [], ls="--", color=INK, lw=0.8, label="decision boundary")]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), bbox_to_anchor=(0.995, 1.0), fontsize=NOTE, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(FIG_DIR / "fig1_overview.png")
    plt.close(fig)


# -------------------------------------------------------------- shared panels
def dose_panel(a, res, methods, key="gap_rms_bc"):
    """One metric vs. bias strength alpha of the training data (line: mean over subsample draws; corrected gap: band of the
    mean 95% test-bootstrap interval)."""
    alphas = np.array(res["alphas"])
    for name in methods:
        label, color, marker, ls, hollow = SERIES[name]
        sel = [[r for r in res["rows"] if r["method"] == name and r["alpha"] == al] for al in alphas]  # (alpha, draw)
        v = np.array([[r[key] for r in rs] for rs in sel]).mean(1) * (100 if key == "acc" else 1)
        if key == "gap_rms_bc":
            ci = np.array([[r["gap_rms_ci"] for r in rs] for rs in sel]).mean(1)
            a.fill_between(alphas, ci[:, 0], ci[:, 1], color=color, alpha=BAND, lw=0)
        a.plot(alphas, v, color=color, marker=marker, ls=ls, label=label, mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE)
    a.set_xticks(alphas, [f"{al:g}" for al in alphas])
    a.set(xlabel="클래스와 그룹 편향 강도 (α)", ylabel="정확도 (%)" if key == "acc" else "보정 GAP")  # no mathtext next to Hangul (no glyph fallback there)


def dose_pair(fig, ax, res, methods):
    """Panels (a) corrected gap and (b) accuracy vs. alpha with a shared legend above the figure."""
    dose_panel(ax[0], res, methods)
    dose_panel(ax[1], res, methods, "acc")
    ax[0].set_title("(a) 재현율 격차")
    ax[1].set_title("(b) 정확도")
    lo, hi = ax[1].get_ylim()
    ax[1].set_ylim(lo - 0.5 * (hi - lo), hi + 0.3 * (hi - lo))  # wider range: differences of a few tenths of a point stay in proportion
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.02, 1.07))


def tradeoff_panel(a, trade, paths, points, from_k=1):
    """Accuracy change vs. bias-corrected gap on the full data (retrained models).  BME paths over the number of removed directions
    (both orders, from the trade-off result); `paths`: further (key, rows, base) paths such as INLP rounds; `points`: (key, row, base)
    single methods.  Accuracy is relative to the original model of the same run; every path starts at the original model.
    from_k: first number of directions drawn on the BME paths (smaller ones coincide with the original model within noise)."""
    base = next(r for r in trade["rows"] if r["method"] == "none")
    path = lambda m: sorted((r for r in trade["rows"] if r["method"] == m and r["k"] >= from_k), key=lambda r: r["k"])  # noqa: E731
    for key, rows, ref in [("bme", path("bme"), base), ("bme-uniform", path("bme-uniform"), base), *paths]:
        label, color, marker, ls, hollow = SERIES[key]
        a.plot([0.0] + [100 * (r["acc"] - ref["acc"]) for r in rows], [ref["gap_rms_bc"]] + [r["gap_rms_bc"] for r in rows], color=color, ls=ls,
               marker=marker, label=label, mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE)
    for key, r, ref in [*points, ("none", base, base)]:
        a.plot(100 * (r["acc"] - ref["acc"]), r["gap_rms_bc"], "o", color=SERIES[key][1], ls="", label=SERIES[key][0])
    a.set(xlabel="정확도 변화 (%p)", ylabel="보정 GAP", title="(b) 정확도와 재현율 격차")


def shared_legend(fig, axes, keys, ncol):
    """One legend above the panels: entries in the order of `keys`, filled column by column (two rows)."""
    found = {}
    for a in axes:
        for handle, label in zip(*a.get_legend_handles_labels()):
            found.setdefault(label, handle)
    labels = [SERIES[k][0] for k in keys]
    fig.legend([found[label] for label in labels], labels, loc="upper left", ncol=ncol, bbox_to_anchor=(0.02, 1.13))


# --------------------------------------------------------------- Figures 2-4
def fig_bios(dose, removal, aug, trade, methods=("none", "leace", "inlp", "bme")):
    """Figure 2 (Bias in Bios): (a) bias-corrected gap vs. bias strength alpha of the training data, (b) accuracy change vs. gap of
    every method on the full data, (c) gap vs. number of removed directions for the bias matrix and the data matrix."""
    with mpl.rc_context(KOREAN_RC):
        fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 1.7), gridspec_kw=WIDE_MIDDLE)
        dose_panel(ax[0], dose, methods)
        ax[0].set_title("(a) 재현율 격차")
        names, data = by_name(removal["rows"]), by_name(aug["rows"])
        tradeoff_panel(ax[1], trade,
                       [("inlp", family(removal["rows"], "inlp")[1], names["none"]), ("inlp-class", family(removal["rows"], "inlp-class")[1], names["none"])],
                       [("leace", names["leace"], names["none"]), ("reweighing", data["reweighing"], data["none"]), ("bma", data["bma"], data["none"]),
                        ("cda", data["cda-all"], data["none"])])
        # Legend Columns: BME | INLP | original, BMA | LEACE, reweighing | CDA
        shared_legend(fig, ax[:2], ["bme", "bme-uniform", "inlp", "inlp-class", "none", "bma", "leace", "reweighing", "cda"], 5)

        # Which Matrix, How Many Directions (full data, retrained model)
        for fam, label, color, marker in [("bme-orth", "bias matrix $B$ (Ours)", PALETTE[0], "o"), ("X-top", "data matrix $X$", PALETTE[3], "o")]:  # amber: green is INLP
            k, sel = family(removal["rows"], fam)
            ci = np.array([r["gap_rms_ci"] for r in sel])
            ax[2].fill_between(k, ci[:, 0], ci[:, 1], color=color, alpha=BAND, lw=0)
            ax[2].plot(k, [r["gap_rms_bc"] for r in sel], color=color, marker=marker, label=label)
        hline(ax[2], names["none"]["gap_rms_bc"], "Original")
        ax[2].set_xscale("log", base=2)
        ax[2].set_xticks([1, 2, 4, 8, 16, 28, 64], ["1", "2", "4", "8", "16", "28", "64"])
        ax[2].minorticks_off()
        ax[2].set(xlabel="제거한 방향의 수 (k)", ylabel="보정 GAP", title="(c) 제거 편향 방향 수")
        ax[2].legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
        finish(fig, "fig2_bios.png")


def fig_bma(dose, aug, methods=("none", "reweighing", "bma")):
    """Figure 3 (Bias in Bios): (a) corrected gap and (b) accuracy vs. alpha for the training-data methods,
    (c) corrected gap vs. amount of added samples on the full data: BMA vs. CDA (gender-swapped text)."""
    with mpl.rc_context(KOREAN_RC):
        fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 1.7))
        dose_pair(fig, ax, dose, methods)

        # Same Cells Filled by the Minimal Plan or for Every Sample (1.0 N)
        data = by_name(aug["rows"])
        for key, part, full in (("bma", "bma", "bma-all"), ("cda", "cda-partial", "cda-all")):
            sel = [data["none"], data[part], data[full]]
            x, ci = [0.0] + [r["added_over_n"] for r in sel[1:]], np.array([r["gap_rms_ci"] for r in sel])
            ax[2].fill_between(x, ci[:, 0], ci[:, 1], color=SERIES[key][1], alpha=BAND, lw=0)
            ax[2].plot(x, [r["gap_rms_bc"] for r in sel], color=SERIES[key][1], marker="o", label=SERIES[key][0])
        ax[2].plot(data["reweighing"]["added_over_n"], data["reweighing"]["gap_rms_bc"], "o", color=SERIES["reweighing"][1], ls="", label="Reweighing")
        ax[2].set(xlabel="추가한 표본의 양 (원본 대비)", ylabel="보정 GAP", title="(c) 증강 데이터 양")
        ax[2].legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
        finish(fig, "fig3_bma.png")


def fig_korea(dose, hazard, trade, methods=("none", "leace", "inlp", "bme", "bma")):
    """Figure 4 (hazard consultations): (a) corrected gap vs. alpha, (b) accuracy change vs. gap of every method on the full data,
    (c) per-item recall difference between the two sales channels vs. the remote-sale share of the item."""
    with mpl.rc_context(KOREAN_RC):
        fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 1.7), gridspec_kw=WIDE_MIDDLE)
        dose_panel(ax[0], dose, methods)
        ax[0].set_title("(a) 재현율 격차")
        names = by_name(hazard["rows"])
        base = next(r for r in trade["rows"] if r["method"] == "none")
        # INLP as One Point: its path stays at the original model.  Zoomed Range: the last steps of BME lie to the left
        inlp = next(r for r in trade["rows"] if r["method"] == "inlp")
        tradeoff_panel(ax[1], trade, [], [("leace", names["leace"], names["none"]), ("inlp", inlp, base),
                                          ("reweighing", names["reweighing"], names["none"]), ("bma", names["bma"], names["none"])], from_k=4)
        ax[1].set(xlim=(-1.3, 0.08), ylim=(0.150, 0.215))
        # Legend Columns: BME | INLP, LEACE | original, BMA | reweighing
        shared_legend(fig, ax[:2], ["bme", "bme-uniform", "inlp", "leace", "none", "bma", "reweighing"], 4)

        # Items with Few Remote Sales Lose Remote-Sale Recall (full data; original model and BME retrained)
        share = np.array(hazard["share"])
        for name in ("none", "bme"):
            label, color = SERIES[name][:2]
            gap = np.array(names[name]["gap"])
            slope, icpt = np.polyfit(share, gap, 1)
            ax[2].plot(share, gap, "o", color=color, ms=3.2, label=label)
            ax[2].plot([0, share.max() * 1.08], [icpt, icpt + slope * share.max() * 1.08], color=color, lw=0.9)
        ax[2].axhline(0, color=AXIS, lw=0.7)
        ax[2].set(xlabel="품목의 비대면 판매 비율", ylabel="재현율 차이 (비대면 − 일반)", title="(c) 품목별 재현율 차이")
        ax[2].legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
        finish(fig, "fig4_korea.png")


def main():
    setup_style()
    mpl.rcParams.update(PAPER_RC)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads(p.read_text(encoding="utf-8")) if (p := RESULTS_DIR / name).exists() else None  # noqa: E731
    spec, removal, aug, hazard = load("spectrum.json"), load("removal.json"), load("augmentation.json"), load("hazard.json")
    dose = {data: load(f"dose_{data}.json") for data in ("bios", "hazard")}
    trade = {data: load(f"tradeoff_{data}.json") for data in ("bios", "hazard")}
    if spec:
        fig_overview(spec)
    if removal and dose["bios"] and aug and trade["bios"]:
        fig_bios(dose["bios"], removal, aug, trade["bios"])
    if aug and dose["bios"]:
        fig_bma(dose["bios"], aug)
    if hazard and dose["hazard"] and trade["hazard"]:
        fig_korea(dose["hazard"], hazard, trade["hazard"])


if __name__ == "__main__":
    main()
