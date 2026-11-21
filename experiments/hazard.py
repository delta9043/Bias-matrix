"""Korean public data: hazard consultations of the 1372 consumer centre, two groups -> hazard.json.

Class: item category (title and body of the case), group: sales channel (general store vs. remote sale).
Table 3 (upper block), Figures 4b-c.
"""
import argparse

import numpy as np

from bias_matrix import (bme_eraser, cell_stats, conditional_bias_matrix, covariance, mean_shift, minimal_plan, mutual_information,
                         permutation_null, plan_sources, sequential_rank)

from . import models as M
from .baselines import leace_eraser, reweighing_weights
from .common import load_task, save_json, selected
from .metrics import group_roles, score, tpr_by_cell

N_PERM = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=M.default_device())
    M.DEVICE = ap.parse_args().device
    layer, l2 = selected()
    rng = np.random.default_rng(0)
    Xtr, ytr, ztr, Xte, yte, zte, C = load_task("hazard", layer)
    means, counts = cell_stats(Xtr, ytr, ztr, C)
    share = counts[:, 1] / counts.sum(1)  # training share of remote sales in each class
    minority, majority = group_roles(counts)
    print(f"layer {layer} | train {Xtr.shape} test {Xte.shape} | classes {C} | group-1 share {ztr.mean():.3f} "
          f"(by class {share.min():.2f} ~ {share.max():.2f}) | device {M.DEVICE}", flush=True)

    rows = []

    def add(name, pred, **info):
        tpr = tpr_by_cell(yte, pred, zte, C)
        rows.append({"method": name} | score(yte, pred, zte, C, 2, minority, majority) | {"gap": tpr[:, 1] - tpr[:, 0]} | info)
        r = rows[-1]
        print(f"{name:11s} acc {r['acc']:.4f} | gap {r['gap_rms']:.4f} (bc {r['gap_rms_bc']:.4f} [{r['gap_rms_ci'][0]:.4f}, {r['gap_rms_ci'][1]:.4f}]) | "
              f"min-TPR {r['tpr_minority']:.4f} maj-TPR {r['tpr_majority']:.4f}", flush=True)

    # Original Model
    base = M.fit_linear(Xtr, ytr, C, l2=l2)
    add("none", base.predict(Xte))

    # Bias Matrix; Number of Significant Directions (sequential permutation test)
    mu, Sigma = covariance(Xtr)
    B = conditional_bias_matrix(Xtr, ytr, ztr, C)
    s = np.linalg.svd(B, compute_uv=False)
    k_perm, _ = sequential_rank(B, permutation_null(Xtr, ytr, ztr, N_PERM, rng, C))
    bias_share = float(np.sum(s**2) / np.trace(Sigma))
    print(f"  bias matrix: tr(BB^T)/tr(Sigma) {bias_share:.5f} | k (permutation) {k_perm} of {C}", flush=True)

    # LEACE (class-free matrix, retrained); BME (oblique, all directions): frozen and retrained
    eraser = leace_eraser(Xtr, ztr, mu, Sigma)
    add("leace", M.fit_linear(eraser(Xtr), ytr, C, l2=l2).predict(eraser(Xte)))
    eraser = bme_eraser(B, mu, Sigma)
    Etr, Ete = eraser(Xtr), eraser(Xte)
    add("bme-frozen", base.predict(Ete))
    add("bme", M.fit_linear(Etr, ytr, C, l2=l2).predict(Ete))
    del Etr, Ete
    M.free_memory()

    # Training-Data Methods: reweighing, BMA on the minimal plan (embeddings untouched)
    add("reweighing", M.fit_linear(Xtr, ytr, C, l2=l2, weight=reweighing_weights(counts)[ytr, ztr]).predict(Xte), added=0.0)
    plan, q_star = minimal_plan(counts)
    src, target = plan_sources(ytr, ztr, plan, rng)
    added = len(src) / len(Xtr)
    extra = mean_shift(Xtr, ytr, ztr, src, target, means)
    add("bma", M.fit_linear(np.concatenate([Xtr, extra]), np.concatenate([ytr, ytr[src]]), C, l2=l2).predict(Xte), added=added)
    print(f"  minimal plan: {len(src)} additions ({added:.3f} n) | target shares {np.round(q_star, 3)}", flush=True)

    save_json("hazard.json", {
        "layer": layer, "l2": l2, "n_train": len(Xtr), "n_test": len(Xte), "n_classes": C, "share": share, "group_share": float(ztr.mean()),
        "k_perm": k_perm, "bias_share_of_variance": bias_share, "mi": mutual_information(counts), "plan_added_over_n": added,
        "q_star": q_star, "rows": rows})


if __name__ == "__main__":
    main()
