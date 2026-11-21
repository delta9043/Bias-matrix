"""Training-data methods on Bias in Bios (full training data) -> augmentation.json.

Reweighing, BMA (mean-shift copies in the cells of the minimal plan), CDA (gender-swapped text) in the same cells
and for every sample.  Embeddings are untouched; only the model is retrained.  Table 2 (lower block), Figures 2b, 3c.
"""
import argparse

import numpy as np

from bias_matrix import cell_stats, mean_shift, minimal_plan, plan_sources

from . import models as M
from .baselines import reweighing_weights
from .common import load_emb, load_task, load_uid, save_json, selected
from .metrics import group_roles, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=M.default_device())
    M.DEVICE = ap.parse_args().device
    layer, l2 = selected()
    rng = np.random.default_rng(0)
    X, y, z, Xte, yte, zte, C = load_task("bios", layer)
    m = len(X)

    # Gender-Swapped Training Texts (CDA), Row-Aligned with the Originals
    uid_cf, Xcf = load_emb("cf", "train", layer)
    assert (uid_cf == load_uid("orig", "train")).all()
    means, counts = cell_stats(X, y, z, C)
    minority, majority = group_roles(counts)
    print(f"layer {layer} | train {X.shape} test {Xte.shape} | device {M.DEVICE}", flush=True)

    rows = []

    def run(name, Xa, ya, weight=None):
        clf = M.fit_linear(Xa, ya, C, l2=l2, weight=weight)
        rows.append({"method": name, "added_over_n": (len(ya) - m) / m} | score(yte, clf.predict(Xte), zte, C, 2, minority, majority))
        r = rows[-1]
        print(f"{name:12s} added {r['added_over_n']:.3f}N | acc {r['acc']:.4f} gap {r['gap_rms']:.4f} (bc {r['gap_rms_bc']:.4f} "
              f"[{r['gap_rms_ci'][0]:.4f}, {r['gap_rms_ci'][1]:.4f}]) min-TPR {r['tpr_minority']:.4f} maj-TPR {r['tpr_majority']:.4f}", flush=True)
        del Xa
        M.free_memory()

    def stack(extra, idx):
        return np.concatenate([X, extra]), np.concatenate([y, y[idx]])

    # Original Model and Reweighing
    run("none", X, y)
    run("reweighing", X, y, weight=reweighing_weights(counts)[y, z])

    # Minimal Plan: cells and sources of the additions
    for _ in range(14):  # random stream of the reported run (noise samples of an earlier version were drawn here)
        rng.standard_normal((m, C), dtype=np.float32)
    plan, q_star = minimal_plan(counts)
    src, target = plan_sources(y, z, plan, rng)
    print(f"minimal plan: {len(src)} additions ({len(src) / m:.3f}N), repeated sources {len(src) - len(np.unique(src))} | q* {np.round(q_star, 3)}",
          flush=True)

    # BMA and CDA in the Cells of the Plan; Both for Every Sample
    everyone = np.arange(m)
    run("bma", *stack(mean_shift(X, y, z, src, target, means), src))
    run("cda-partial", *stack(Xcf[src], src))
    run("cda-all", *stack(Xcf, everyone))
    run("bma-all", *stack(mean_shift(X, y, z, everyone, 1 - z, means), everyone))

    save_json("augmentation.json", {"layer": layer, "l2": l2, "n_train": m, "n_test": len(Xte), "q_star": q_star, "n_additions": len(src),
                                    "rows": rows})


if __name__ == "__main__":
    main()
