"""Cell statistics and covariance helpers.

Conventions: X is (n, d) with samples in rows; class y in {0..C-1}; group z in {0..G-1}.
A cell is one (class, group) pair.
"""
import numpy as np
import scipy.sparse as sp


def table_shape(y, z, n_classes=None, n_groups=None):
    """Number of classes and groups; inferred from the labels when not given."""
    return (int(y.max()) + 1 if n_classes is None else n_classes), (int(z.max()) + 1 if n_groups is None else n_groups)


def count_table(y, z, n_classes=None, n_groups=None):
    """Counts (C, G) of the (class, group) cells."""
    C, G = table_shape(y, z, n_classes, n_groups)
    return np.bincount(y * G + z, minlength=C * G).reshape(C, G).astype(np.float64)


def cell_stats(X, y, z, n_classes=None, n_groups=None):
    """Means (C, G, d) and counts (C, G) of the (class, group) cells."""
    C, G = table_shape(y, z, n_classes, n_groups)
    cell = y * G + z
    n = len(cell)
    # One Sparse Product for All Cell Sums (accumulated in the dtype of X)
    ind = sp.csr_matrix((np.ones(n, dtype=X.dtype), (cell, np.arange(n))), shape=(C * G, n))
    counts = np.bincount(cell, minlength=C * G).astype(np.float64)
    mu = np.asarray(ind @ X, dtype=np.float64) / np.maximum(counts, 1)[:, None]
    return mu.reshape(C, G, -1), counts.reshape(C, G)


def covariance(X, chunk=32768):
    """Mean (d,) and covariance (d, d); centered Gram accumulated in float64 (whitening is precision-sensitive)."""
    mu = X.mean(0, dtype=np.float64)
    G = np.zeros((X.shape[1], X.shape[1]))
    for s in range(0, len(X), chunk):
        Xc = X[s : s + chunk].astype(np.float64) - mu
        G += Xc.T @ Xc
    return mu, G / len(X)


def sym_power(S, p, rtol=1e-10):
    """Symmetric PSD matrix power S^p (pseudo-inverse sense for p < 0)."""
    w, V = np.linalg.eigh(S)
    keep = w > rtol * w.max()
    return (V[:, keep] * w[keep] ** p) @ V[:, keep].T


def mutual_information(counts):
    """I(Y; Z) in nats of a (C, G) count table."""
    p = counts / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p / (p.sum(1, keepdims=True) * p.sum(0, keepdims=True)))
    return float(np.where(p > 0, p * pmi, 0.0).sum())
