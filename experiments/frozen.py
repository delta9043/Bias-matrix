"""BME on an already trained model (Bias in Bios, full training data) -> frozen.json.

The original model is kept and only the embeddings are edited (no retraining).  Table 2, row "BME (Ours, 모델 고정)".
"""
import argparse

from bias_matrix import bme_eraser, conditional_bias_matrix, count_table, covariance

from . import models as M
from .common import load_task, save_json, selected
from .metrics import group_roles, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=M.default_device())
    M.DEVICE = ap.parse_args().device
    layer, l2 = selected()
    Xtr, ytr, ztr, Xte, yte, zte, C = load_task("bios", layer)
    minority, majority = group_roles(count_table(ytr, ztr, C))
    print(f"layer {layer} | train {Xtr.shape} test {Xte.shape} | device {M.DEVICE}", flush=True)

    # Original Model; Oblique BME (all directions) Applied to the Test Embeddings Only
    base = M.fit_linear(Xtr, ytr, C, l2=l2)
    mu, Sigma = covariance(Xtr)
    eraser = bme_eraser(conditional_bias_matrix(Xtr, ytr, ztr, C), mu, Sigma)
    rows = []
    for name, X in (("none", Xte), ("bme-frozen", eraser(Xte))):
        rows.append({"method": name} | score(yte, base.predict(X), zte, C, 2, minority, majority))
        r = rows[-1]
        print(f"{name:11s} acc {r['acc']:.4f} gap {r['gap_rms']:.4f} (bc {r['gap_rms_bc']:.4f} [{r['gap_rms_ci'][0]:.4f}, {r['gap_rms_ci'][1]:.4f}]) "
              f"min-TPR {r['tpr_minority']:.4f} maj-TPR {r['tpr_majority']:.4f}", flush=True)
    save_json("frozen.json", {"layer": layer, "l2": l2, "n_train": len(Xtr), "n_test": len(Xte), "rows": rows})


if __name__ == "__main__":
    main()
