"""Accuracy vs. recall gap of BME by the number of removed directions (full training data) -> tradeoff_{data}.json.

Two orders of the directions: the bias matrix B (cell-size weights, method "bme") and class-uniform columns
("bme-uniform"); both remove the same space at full rank.  Models are retrained.  Figures 2b, 4b.
--inlp: INLP as well (data whose INLP row is in no other result file).
"""
import argparse
import time

import numpy as np

from bias_matrix import bme_eraser, conditional_bias_matrix, count_table, covariance

from . import models as M
from .baselines import inlp_erasers
from .common import load_task, save_json, selected
from .metrics import group_roles, score

K_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)
INLP_ROUNDS = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bios", choices=["bios", "hazard"])
    ap.add_argument("--inlp", action="store_true")
    ap.add_argument("--device", default=M.default_device())
    args = ap.parse_args()
    M.DEVICE = args.device
    layer, l2 = selected()
    Xtr, ytr, ztr, Xte, yte, zte, C = load_task(args.data, layer)
    minority, majority = group_roles(count_table(ytr, ztr, C))
    ks = [k for k in K_GRID if k < C] + [C]  # C: all directions (same edit for both orders)
    print(f"{args.data} | layer {layer} | train {Xtr.shape} test {Xte.shape} | classes {C} | k {ks} | device {M.DEVICE}", flush=True)
    rows, t0 = [], time.time()

    def add(method, k, pred):
        rows.append({"method": method, "k": k} | score(yte, pred, zte, C, 2, minority, majority))
        r = rows[-1]
        print(f"{method:11s} k {k:2d} | acc {r['acc']:.4f} | gap {r['gap_rms']:.4f} (bc {r['gap_rms_bc']:.4f}) | "
              f"min-TPR {r['tpr_minority']:.4f} maj-TPR {r['tpr_majority']:.4f} | {time.time() - t0:.0f}s", flush=True)

    # Original Model
    add("none", 0, M.fit_linear(Xtr, ytr, C, l2=l2).predict(Xte))
    mu, Sigma = covariance(Xtr)

    # Oblique BME of the Top-k Directions, Model Retrained
    for method, weights in (("bme", "cell"), ("bme-uniform", "uniform")):
        B = conditional_bias_matrix(Xtr, ytr, ztr, C, weights=weights)
        for k in ks:
            eraser = bme_eraser(B, mu, Sigma, k=k)
            Etr, Ete = eraser(Xtr), eraser(Xte)
            add(method, k, M.fit_linear(Etr, ytr, C, l2=l2).predict(Ete))
            del Etr, Ete
            M.free_memory()

    # INLP: rounds of a class-free group classifier, model retrained (k = number of rounds)
    if args.inlp:
        eraser = inlp_erasers(Xtr, ztr, mu, np.random.default_rng(1000), [INLP_ROUNDS])[INLP_ROUNDS]
        Etr, Ete = eraser(Xtr), eraser(Xte)
        add("inlp", INLP_ROUNDS, M.fit_linear(Etr, ytr, C, l2=l2).predict(Ete))

    save_json(f"tradeoff_{args.data}.json", {"data": args.data, "layer": layer, "l2": l2, "n_train": len(Xtr), "n_test": len(Xte),
                                             "n_classes": C, "ks": ks, "rows": rows})


if __name__ == "__main__":
    main()
