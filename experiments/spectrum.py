"""Bias spectrum of Bias in Bios and the number of directions k by the sequential permutation test -> spectrum.json.

Figure 1 (Step 4) and the selected k of Section 5.2.
"""
import numpy as np

from bias_matrix import covariance, select_rank

from .common import load_task, save_json, selected

N_PERM = 200


def main():
    layer, _ = selected()
    X, y, z, _, _, _, C = load_task("bios", layer)
    k, sv, thr = select_rank(X, y, z, N_PERM, rng=np.random.default_rng(0), n_classes=C)
    _, Sigma = covariance(X)
    share = float((sv**2).sum() / np.trace(Sigma))  # covariance share of the group differences inside the classes
    print(f"layer {layer}: n={len(X)} d={X.shape[1]} | k (sequential permutation test, {N_PERM} shuffles) {k} of {len(sv)} | "
          f"tr(BB^T)/tr(Sigma) {share:.3%}")
    print(f"  s:   {np.round(sv, 3)}\n  thr: {np.round(thr, 3)}")
    save_json("spectrum.json", {"layer": layer, "n": len(X), "d": X.shape[1], "n_perm": N_PERM, "k": k, "sv": sv, "thr": thr,
                                "bias_share_of_variance": share})


if __name__ == "__main__":
    main()
