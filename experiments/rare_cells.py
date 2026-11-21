"""Recall of the rare cells on the hazard consultations (full training data) -> rare_cells.json.

Rare cell: a group whose training share in its class is below 10% (remote sales of items mostly sold in stores).
Pooled recall of the test rows of these cells and of the other group in the same classes.  Section 5.4.
"""
import argparse

import numpy as np

from bias_matrix import bme_eraser, cell_stats, conditional_bias_matrix, covariance, mean_shift, minimal_plan, plan_sources

from . import models as M
from .common import load_task, save_json, selected

RARE = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=M.default_device())
    M.DEVICE = ap.parse_args().device
    layer, l2 = selected()
    rng = np.random.default_rng(0)
    Xtr, ytr, ztr, Xte, yte, zte, C = load_task("hazard", layer)
    means, counts = cell_stats(Xtr, ytr, ztr, C)
    share = counts[:, 1] / counts.sum(1)

    # Test Rows of the Rare Cells and of the Other Group in the Same Classes
    rare = ((share[yte] < RARE) & (zte == 1)) | ((share[yte] > 1 - RARE) & (zte == 0))
    rare_other = ((share[yte] < RARE) | (share[yte] > 1 - RARE)) & ~rare
    print(f"layer {layer} | train {Xtr.shape} test {Xte.shape} | classes {C} | rare-cell rows {rare.sum()} (other group {rare_other.sum()}) | "
          f"device {M.DEVICE}", flush=True)

    rows = []

    def add(name, pred):
        rows.append({"method": name, "acc": float((pred == yte).mean()), "recall_rare": float((pred[rare] == yte[rare]).mean()),
                     "recall_rare_other": float((pred[rare_other] == yte[rare_other]).mean())})
        r = rows[-1]
        print(f"{name:5s} acc {r['acc']:.4f} | rare {r['recall_rare']:.4f} (other {r['recall_rare_other']:.4f})", flush=True)

    # Original Model; BME (oblique, all directions, retrained); BMA on the minimal plan
    add("none", M.fit_linear(Xtr, ytr, C, l2=l2).predict(Xte))
    mu, Sigma = covariance(Xtr)
    eraser = bme_eraser(conditional_bias_matrix(Xtr, ytr, ztr, C), mu, Sigma)
    Etr, Ete = eraser(Xtr), eraser(Xte)
    add("bme", M.fit_linear(Etr, ytr, C, l2=l2).predict(Ete))
    del Etr, Ete
    M.free_memory()
    plan, _ = minimal_plan(counts)
    src, target = plan_sources(ytr, ztr, plan, rng)
    extra = mean_shift(Xtr, ytr, ztr, src, target, means)
    add("bma", M.fit_linear(np.concatenate([Xtr, extra]), np.concatenate([ytr, ytr[src]]), C, l2=l2).predict(Xte))

    save_json("rare_cells.json", {"layer": layer, "l2": l2, "n_train": len(Xtr), "n_test": len(Xte), "n_classes": C, "rare_share": RARE,
                                  "n_rare": int(rare.sum()), "n_rare_other": int(rare_other.sum()), "rows": rows})


if __name__ == "__main__":
    main()
