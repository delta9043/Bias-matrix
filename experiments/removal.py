"""Embedding edits on Bias in Bios (full training data) -> removal.json.

Which matrix (bias matrix B, data matrix X, class-free baselines), how many directions, how strongly.
Table 2 (upper block), Figure 2b (LEACE, INLP paths), Figure 2c.
"""
import argparse
import time

import numpy as np

from bias_matrix import bme_eraser, conditional_bias_matrix, count_table, covariance

from . import models as M
from .baselines import inlp_erasers, leace_eraser, top_pc_eraser
from .common import RESULTS_DIR, load_json, load_task, save_json, selected
from .metrics import gap_sq, group_roles, score, tpr_by_cell

K_ORTH = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28]  # orthogonal BME path (plus the k of the permutation test)
K_TOP_PC = [1, 2, 4, 8, 16, 28, 64]
SOFT = 0.25  # soft editing: singular values of the whitened B times this factor
INLP_ROUNDS = [4, 8, 16, 32]
INLP_CLASS_ROUNDS = [8, 16, 32, 64, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=M.default_device())
    M.DEVICE = ap.parse_args().device
    layer, l2 = selected()
    rng = np.random.default_rng(0)
    Xtr, ytr, ztr, Xte, yte, zte, C = load_task("bios", layer)
    minority, majority = group_roles(count_table(ytr, ztr, C))
    print(f"layer {layer} | train {Xtr.shape} test {Xte.shape} | device {M.DEVICE}", flush=True)

    # Training Statistics, Bias Matrix, Original Model (also the frozen model of every edit)
    mu, Sigma = covariance(Xtr)
    B = conditional_bias_matrix(Xtr, ytr, ztr, C)
    base = M.fit_linear(Xtr, ytr, C, l2=l2)
    d = Xtr.shape[1]

    # k of the Sequential Permutation Test Joins the Orthogonal Path
    k_sel = load_json("spectrum.json")["k"] if (RESULTS_DIR / "spectrum.json").exists() else None
    k_orth = sorted(set(K_ORTH) | ({k_sel} if k_sel else set()))

    # Method Grid: (name, family, parameter, eraser)
    methods = [("none", "none", 0, None), ("leace", "leace", 1, leace_eraser(Xtr, ztr, mu, Sigma))]
    methods += [(f"bme-orth-{k}", "bme-orth", k, bme_eraser(B, mu, k=k)) for k in k_orth]
    methods += [("bme", "bme", C, bme_eraser(B, mu, Sigma)), (f"bme-soft-{SOFT}", "bme-soft", SOFT, bme_eraser(B, mu, Sigma, shrink=SOFT))]
    methods += [(f"X-top-{k}", "X-top", k, top_pc_eraser(Sigma, mu, k)) for k in K_TOP_PC]
    rng.standard_normal(87 * d)  # random stream of the reported run (control rows of an earlier version drew here)
    methods += [(f"inlp-{t}", "inlp", t, er) for t, er in inlp_erasers(Xtr, ztr, mu, rng, INLP_ROUNDS).items()]
    methods += [(f"inlp-class-{t}", "inlp-class", t, er)
                for t, er in inlp_erasers(Xtr, ztr, mu, rng, INLP_CLASS_ROUNDS, y=ytr, n_classes=C).items()]

    rows = []
    for name, family, param, eraser in methods:
        t0 = time.time()
        Etr, Ete = (Xtr, Xte) if eraser is None else (eraser(Xtr), eraser(Xte))
        # Frozen Model: the original model on the edited test embeddings
        frozen = base.predict(Ete)
        row = {"method": name, "family": family, "param": param, "frozen_acc": float((frozen == yte).mean()),
               "frozen_gap_rms": float(np.sqrt(gap_sq(tpr_by_cell(yte, frozen, zte, C))))}
        # Retrained Model: trained on the edited training embeddings
        clf = base if eraser is None else M.fit_linear(Etr, ytr, C, l2=l2)
        row |= score(yte, clf.predict(Ete), zte, C, 2, minority, majority)
        rows.append(row)
        print(f"{name:16s} frozen acc {row['frozen_acc']:.4f} gap {row['frozen_gap_rms']:.4f} | retrained acc {row['acc']:.4f} "
              f"gap {row['gap_rms']:.4f} (bc {row['gap_rms_bc']:.4f} [{row['gap_rms_ci'][0]:.4f}, {row['gap_rms_ci'][1]:.4f}]) "
              f"min-TPR {row['tpr_minority']:.4f} maj-TPR {row['tpr_majority']:.4f} | {time.time() - t0:.0f}s", flush=True)
        del Etr, Ete
        M.free_memory()

    save_json("removal.json", {"layer": layer, "l2": l2, "n_train": len(Xtr), "n_test": len(Xte), "n_classes": C, "k_selected": k_sel, "rows": rows})


if __name__ == "__main__":
    main()
