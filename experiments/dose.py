"""Bias strength of the training data vs. the recall gap of the model -> dose_{data}.json.

Class sizes are fixed and only the group shares inside the classes change: alpha = 0 (class and group independent)
to alpha = 1 (original shares).  Every method is trained on the same subsample.  Figures 2a, 3a-b, 4a.
"""
import argparse
import time

import numpy as np

from bias_matrix import (bme_eraser, cell_stats, conditional_bias_matrix, count_table, covariance, mean_shift, minimal_plan,
                         mutual_information, plan_sources)

from . import models as M
from .baselines import inlp_erasers, leace_eraser, reweighing_weights
from .common import load_task, save_json, selected
from .metrics import group_roles, score

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# none: original model; leace: class-free bias matrix, oblique; inlp: INLP_ROUNDS rounds of a class-free group classifier;
# bme: oblique BME, all directions, bias matrix of the whole training set; bma, reweighing: training-data methods
METHODS = ["none", "leace", "inlp", "bme", "bma", "reweighing"]
INLP_ROUNDS = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bios", choices=["bios", "hazard"])
    ap.add_argument("--seeds", type=int, default=5)  # independent draws of the nested subsamples
    ap.add_argument("--device", default=M.default_device())
    args = ap.parse_args()
    M.DEVICE = args.device
    layer, l2 = selected()
    Xtr, ytr, ztr, Xte, yte, zte, C = load_task(args.data, layer)

    # Fixed Class Sizes for Every Alpha: the largest subsample that still allows equal group shares (alpha = 0)
    n_tab = count_table(ytr, ztr, C)
    share = n_tab[:, 1] / n_tab.sum(1)  # original share of group 1 in each class
    q = float(ztr.mean())
    m = np.floor(np.minimum(n_tab[:, 1] / q, n_tab[:, 0] / (1 - q))).astype(int)
    minority, majority = group_roles(n_tab)
    print(f"{args.data} | layer {layer} | train {Xtr.shape} test {Xte.shape} | classes {C} | group-1 share {q:.3f} "
          f"(by class {share.min():.2f} ~ {share.max():.2f}) | subsample {m.sum()} | seeds {args.seeds} | device {M.DEVICE}", flush=True)

    # BME Fitted Once on the Whole Training Set
    mu_all, Sigma_all = covariance(Xtr)
    bme_all = bme_eraser(conditional_bias_matrix(Xtr, ytr, ztr, C), mu_all, Sigma_all)
    Ete_all = bme_all(Xte)

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        order = {(c, g): rng.permutation(np.flatnonzero((ytr == c) & (ztr == g))) for c in range(C) for g in (0, 1)}  # nested subsamples
        for alpha in ALPHAS:
            t0 = time.time()
            # Training Subset with Group Shares q + alpha (share - q) in Every Class
            n1 = np.round(m * (q + alpha * (share - q))).astype(int)
            idx = np.concatenate([np.r_[order[c, 1][: n1[c]], order[c, 0][: m[c] - n1[c]]] for c in range(C)])
            X, y, z = Xtr[idx], ytr[idx], ztr[idx]
            tab = count_table(y, z, C)
            mi = mutual_information(tab)
            mu, Sigma = covariance(X)
            preds, added = {}, 0.0

            # Original Model; Class-Free Baselines
            preds["none"] = M.fit_linear(X, y, C, l2=l2).predict(Xte)
            er = leace_eraser(X, z, mu, Sigma)
            preds["leace"] = M.fit_linear(er(X), y, C, l2=l2).predict(er(Xte))
            er = inlp_erasers(X, z, mu, np.random.default_rng(1000 + seed), [INLP_ROUNDS])[INLP_ROUNDS]  # own generator: BMA draws unaffected
            preds["inlp"] = M.fit_linear(er(X), y, C, l2=l2).predict(er(Xte))

            # BME: model retrained on the edited subset
            preds["bme"] = M.fit_linear(bme_all(X), y, C, l2=l2).predict(Ete_all)

            # BMA and Reweighing on the Same Subset
            if alpha == 0:  # independent table: no additions, unit weights
                preds["bma"] = preds["reweighing"] = preds["none"]
            else:
                plan, _ = minimal_plan(tab)
                src, target = plan_sources(y, z, plan, rng)
                added = len(src) / len(idx)
                extra = mean_shift(X, y, z, src, target, cell_stats(X, y, z, C)[0])
                preds["bma"] = M.fit_linear(np.concatenate([X, extra]), np.concatenate([y, y[src]]), C, l2=l2).predict(Xte)
                del extra
                preds["reweighing"] = M.fit_linear(X, y, C, l2=l2, weight=reweighing_weights(tab)[y, z]).predict(Xte)

            for name in METHODS:
                rows.append({"method": name, "seed": seed, "alpha": alpha, "n": len(idx), "mi": mi, "added": added if name == "bma" else 0.0}
                            | score(yte, preds[name], zte, C, 2, minority, majority))
                r = rows[-1]
                print(f"seed {seed} alpha {alpha:.2f} MI {mi:.4f} | {name:10s} acc {r['acc']:.4f} gap bc {r['gap_rms_bc']:.4f} "
                      f"[{r['gap_rms_ci'][0]:.4f}, {r['gap_rms_ci'][1]:.4f}] min-TPR {r['tpr_minority']:.4f} maj-TPR {r['tpr_majority']:.4f}", flush=True)
            print(f"  seed {seed} alpha {alpha:.2f} done in {time.time() - t0:.0f}s", flush=True)
            del X, preds
            M.free_memory()

    save_json(f"dose_{args.data}.json", {
        "data": args.data, "layer": layer, "l2": l2, "n_test": len(Xte), "n_classes": C, "share": share, "q": q,
        "class_sizes": m, "alphas": ALPHAS, "seeds": args.seeds, "rows": rows})


if __name__ == "__main__":
    main()
