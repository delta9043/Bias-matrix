"""Conditional bias matrix and Bias-Matrix Editing (BME): Sections 3.2-3.3 and Algorithm 1 of the paper.

A bias matrix B is (d, C G); its column space holds the directions along which the groups differ inside the classes.
"""
import numpy as np

from .stats import cell_stats, covariance, sym_power


# ---------------------------------------------------------------- bias matrix
def conditional_bias_matrix(X, y, z, n_classes=None, n_groups=None, weights="cell"):
    """B (d, C G) with one column per (class, group) cell: b_zy = sqrt(n_zy / n) (mu_zy - mu_y).

    B B^T is the part of the covariance explained by the group inside the classes.
    weights="uniform": columns mu_zy - (plain average of the group means of class y), no cell-size weights
    (same column space, other order of the singular directions).
    """
    if weights not in ("cell", "uniform"):
        raise ValueError(f"weights must be 'cell' or 'uniform', got {weights!r}")
    mu, counts = cell_stats(X, y, z, n_classes, n_groups)
    if weights == "cell":
        center = (mu * counts[..., None]).sum(1) / np.maximum(counts.sum(1), 1)[:, None]  # class means mu_y
        scale = np.sqrt(counts / len(X))
    else:
        scale = (counts > 0).astype(np.float64)  # empty cells give no column
        center = (mu * scale[..., None]).sum(1) / np.maximum(scale.sum(1), 1)[:, None]
    return ((mu - center[:, None]) * scale[..., None]).reshape(-1, mu.shape[-1]).T


def numerical_rank(s, rtol=1e-8):
    """Number of singular values above rtol times the largest (s sorted in decreasing order)."""
    return int((s > rtol * s[0]).sum())


# -------------------------------------------------------------------- erasers
class Eraser:
    """Affine map x' = x - L R^T (x - mean); L, R are (d, k)."""

    def __init__(self, L, R, mean):
        self.L, self.R, self.mean = L, R, mean

    def __call__(self, X):
        coef = X @ self.R.astype(X.dtype) - (self.mean @ self.R).astype(X.dtype)
        return X - coef @ self.L.T.astype(X.dtype)

    def matrix(self):
        """Linear part A (d, d): x' = mean + A (x - mean)."""
        return np.eye(len(self.mean)) - self.L @ self.R.T


def bme_eraser(B, mean, cov=None, k=None, shrink=None):
    """Eraser that edits the top-k singular values of the bias matrix: s_j -> shrink_j s_j.

    cov=None: orthogonal edit x' = x - U_k (I - diag(shrink)) U_k^T (x - mean), B = U diag(s) V^T.
    cov given: oblique edit, the same projection in the cov^{-1/2}-whitened space (LEACE-type map: least change of x
    in the cov^{-1} norm); the edited singular values are those of cov^{-1/2} B.
    k=None: all directions of B (its numerical rank).  shrink=None: hard edit (0); 0 < shrink < 1: soft edit.
    """
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    rank = numerical_rank(s)  # rank of B and of the whitened B: at most C (G - 1)
    k = rank if k is None else min(k, rank)
    lam = np.zeros(k) if shrink is None else np.broadcast_to(shrink, (k,))
    if cov is None:
        return Eraser(U[:, :k] * (1 - lam), U[:, :k], mean)
    W, W_inv = sym_power(cov, -0.5), sym_power(cov, 0.5)
    Q = np.linalg.svd(W @ B, full_matrices=False)[0][:, :k]
    return Eraser(W_inv @ Q * (1 - lam), W.T @ Q, mean)


def bme(X, y, z, k=None, oblique=True, shrink=None, n_classes=None, n_groups=None):
    """Bias-Matrix Editing fitted on training embeddings (Algorithm 1); returns the eraser.

    eraser(X) edits any embeddings (training and test) with the same map.
    oblique=True (default of the experiments): edit in the whitened space; False: orthogonal projection of Algorithm 1.
    """
    B = conditional_bias_matrix(X, y, z, n_classes, n_groups)
    if not oblique:
        return bme_eraser(B, X.mean(0, dtype=np.float64), None, k, shrink)
    mean, cov = covariance(X)
    return bme_eraser(B, mean, cov, k, shrink)


# ------------------------------------------------------------- rank selection
def shuffle_within(z, y, rng):
    """Permutation of z inside each class (null: z independent of x given y)."""
    zp = z.copy()
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        zp[idx] = z[idx][rng.permutation(len(idx))]
    return zp


def permutation_null(X, y, z, n_perm, rng, n_classes=None, n_groups=None):
    """Bias matrices under within-class shuffles of z: (n_perm, d, C G)."""
    return np.stack([conditional_bias_matrix(X, y, shuffle_within(z, y, rng), n_classes, n_groups) for _ in range(n_perm)])


def sequential_rank(B, null_B, alpha=0.05):
    """Sequential permutation test for the number of directions of B; returns (k, thresholds).

    Direction j is kept if s_j exceeds the (1 - alpha) null quantile of the TOP singular value
    after projecting u_1..u_{j-1} out of the permuted matrices; stops at the first failure.
    (Comparing s_j with the j-th null singular value over-selects: once j - 1 slots are taken by
    signal, s_j competes with the largest remaining noise value, not the j-th one.)
    """
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    rank = numerical_rank(s)
    thr = np.zeros(rank)
    for j in range(rank):
        resid = null_B - U[:, :j] @ (U[:, :j].T @ null_B)
        thr[j] = np.quantile(np.linalg.svd(resid, compute_uv=False)[:, 0], 1 - alpha)
    fail = np.flatnonzero(s[:rank] <= thr)
    return (int(fail[0]) if len(fail) else rank), thr


def select_rank(X, y, z, n_perm=200, alpha=0.05, rng=None, n_classes=None, n_groups=None):
    """Number of directions k by the sequential permutation test: (k, singular values of B, thresholds)."""
    rng = np.random.default_rng(rng)
    B = conditional_bias_matrix(X, y, z, n_classes, n_groups)
    k, thr = sequential_rank(B, permutation_null(X, y, z, n_perm, rng, n_classes, n_groups), alpha)
    return k, np.linalg.svd(B, compute_uv=False)[: len(thr)], thr
