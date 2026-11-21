"""Comparison methods: LEACE (class-free bias matrix), INLP and its by-class variant, top principal components, reweighing."""
import numpy as np

from bias_matrix import Eraser, bme_eraser, conditional_bias_matrix

from .models import fit_linear


def leace_eraser(X, z, mean, cov):
    """LEACE: oblique edit with the class-free bias matrix (group means around the overall mean)."""
    return bme_eraser(conditional_bias_matrix(X, np.zeros(len(z), dtype=int), z), mean, cov)


def top_pc_eraser(cov, mean, k):
    """Control: zero the top-k singular values of the centered data matrix X (top-k principal components)."""
    _, V = np.linalg.eigh(cov)
    U = V[:, ::-1][:, :k]
    return Eraser(U, U, mean)


def inlp_erasers(X, z, mean, rng, rounds, y=None, n_classes=None, sub=50000):
    """INLP: repeatedly fit a linear group classifier and project out its normal direction; {rounds: eraser}.

    y given: by-class variant (classifier fit on one randomly chosen class per round).
    """
    dirs, out = [], {}
    for t in range(1, max(rounds) + 1):
        idx = np.arange(len(X)) if y is None else np.flatnonzero(y == rng.integers(n_classes))
        idx = rng.choice(idx, min(sub, len(idx)), replace=False)
        Xs = Eraser(np.stack(dirs, 1), np.stack(dirs, 1), mean)(X[idx]) if dirs else X[idx]
        clf = fit_linear(Xs, z[idx], 2)
        w = (clf.W[:, 1] - clf.W[:, 0]).astype(np.float64)
        for u in dirs:  # Gram-Schmidt against earlier directions
            w -= (w @ u) * u
        dirs.append(w / np.linalg.norm(w))
        if t in rounds:
            U = np.stack(dirs, 1)
            out[t] = Eraser(U, U, mean)
    return out


def reweighing_weights(counts):
    """Kamiran-Calders cell weights P(y) P(z) / P(y, z) of a (C, G) count table."""
    p = counts / counts.sum()
    return p.sum(1, keepdims=True) * p.sum(0, keepdims=True) / p
