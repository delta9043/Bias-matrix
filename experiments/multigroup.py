"""Korean public data with three groups -> multigroup.json.

Hazard consultations by sales method (general store, online, TV home shopping): the same bias matrix, BME,
minimal plan and BMA with G = 3.  Table 3 (lower block).
"""
import argparse

import numpy as np

from bias_matrix import (bme_eraser, cell_stats, conditional_bias_matrix, covariance, mean_shift, minimal_plan, mutual_information,
                         numerical_rank, permutation_null, plan_sources, sequential_rank)

from . import models as M
from .baselines import reweighing_weights
from .common import load_emb, save_json, selected
from .data import load_hazard
from .metrics import group_roles, score

CODES = ["10", "24", "27"]  # sales-method codes: store, online, TV home shopping (24 and 27 read from body keywords)
MIN_CELL = 400  # rows of every (class, group) cell over train and test
N_PERM = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=M.default_device())
    M.DEVICE = ap.parse_args().device
    layer, l2 = selected()
    rng = np.random.default_rng(0)
    G = len(CODES)

    # Classes with Enough Rows in All Three Groups; Embeddings of the Hazard Run
    df = load_hazard()
    df["grp"] = df.method.map({c: i for i, c in enumerate(CODES)})
    tab = df.groupby(["cls", "grp"]).size().unstack(fill_value=0)
    keep = tab.index[(tab >= MIN_CELL).all(axis=1)].to_numpy()
    remap = np.full(int(df.cls.max()) + 1, -1)
    remap[keep] = np.arange(len(keep))
    C = len(keep)
    cls, grp = remap[df.cls.to_numpy()], df.grp.to_numpy()

    def split(name):
        uid, X = load_emb("hazard", name, layer)
        sel = cls[uid] >= 0
        return X[sel], cls[uid][sel], grp[uid][sel]

    (Xtr, ytr, ztr), (Xte, yte, zte) = split("train"), split("test")
    means, counts = cell_stats(Xtr, ytr, ztr, C, G)
    minority, majority = group_roles(counts)
    codes = df.drop_duplicates("cls").set_index("cls").code[keep].tolist()
    print(f"layer {layer} | train {Xtr.shape} test {Xte.shape} | classes {C} {codes} | groups {G} | group shares {np.round(counts.sum(0) / counts.sum(), 3)} | "
          f"smallest test cell {np.bincount(yte * G + zte).min()} | I(Y;Z) {mutual_information(counts):.4f} | device {M.DEVICE}", flush=True)

    rows = []

    def add(name, pred, **info):
        rows.append({"method": name} | score(yte, pred, zte, C, G, minority, majority) | info)
        r = rows[-1]
        print(f"{name:11s} acc {r['acc']:.4f} | gap {r['gap_rms']:.4f} (bc {r['gap_rms_bc']:.4f} [{r['gap_rms_ci'][0]:.4f}, {r['gap_rms_ci'][1]:.4f}]) | "
              f"min-TPR {r['tpr_minority']:.4f} maj-TPR {r['tpr_majority']:.4f}", flush=True)

    # Original Model
    base = M.fit_linear(Xtr, ytr, C, l2=l2)
    add("none", base.predict(Xte))

    # Bias Matrix (C G columns, rank at most C (G - 1)); Number of Significant Directions
    mu, Sigma = covariance(Xtr)
    B = conditional_bias_matrix(Xtr, ytr, ztr, C, G)
    s = np.linalg.svd(B, compute_uv=False)
    k_perm, _ = sequential_rank(B, permutation_null(Xtr, ytr, ztr, N_PERM, rng, C, G))
    rank, bias_share = numerical_rank(s), float(np.sum(s**2) / np.trace(Sigma))
    print(f"  bias matrix: {B.shape[1]} columns, rank {rank} | tr(BB^T)/tr(Sigma) {bias_share:.5f} | k (permutation) {k_perm}", flush=True)

    # BME (oblique, all directions): frozen and retrained
    eraser = bme_eraser(B, mu, Sigma)
    Etr, Ete = eraser(Xtr), eraser(Xte)
    add("bme-frozen", base.predict(Ete))
    add("bme", M.fit_linear(Etr, ytr, C, l2=l2).predict(Ete))
    del Etr, Ete
    M.free_memory()

    # Training-Data Methods: reweighing, BMA on the minimal plan (embeddings untouched)
    add("reweighing", M.fit_linear(Xtr, ytr, C, l2=l2, weight=reweighing_weights(counts)[ytr, ztr]).predict(Xte), added=0.0)
    plan, q_star = minimal_plan(counts)
    added = plan.sum() / len(Xtr)
    print(f"  minimal plan: {plan.sum()} additions ({added:.3f} n) | target shares {np.round(q_star, 3)} | "
          f"I(Y;Z) after {mutual_information(counts + plan):.6f}", flush=True)
    for c, g in zip(*np.nonzero(plan)):  # random stream of the reported run (a control of an earlier version drew from each cell here)
        n_cell, k = int(counts[c, g]), int(plan[c, g])
        if k <= n_cell:
            rng.choice(n_cell, k, replace=False)
        else:
            rng.choice(n_cell, k - n_cell)
    src, target = plan_sources(ytr, ztr, plan, rng)
    extra = mean_shift(Xtr, ytr, ztr, src, target, means)
    add("bma", M.fit_linear(np.concatenate([Xtr, extra]), np.concatenate([ytr, ytr[src]]), C, l2=l2).predict(Xte), added=added)

    save_json("multigroup.json", {
        "layer": layer, "l2": l2, "group_codes": CODES, "class_codes": codes, "n_train": len(Xtr), "n_test": len(Xte), "n_classes": C, "n_groups": G,
        "counts": counts, "group_shares": counts.sum(0) / counts.sum(), "mi": mutual_information(counts), "rank": rank, "k_perm": k_perm,
        "bias_share_of_variance": bias_share, "plan": plan, "plan_added_over_n": added, "q_star": q_star, "rows": rows})


if __name__ == "__main__":
    main()
