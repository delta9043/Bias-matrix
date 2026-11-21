"""Bias-Matrix Augmentation (BMA): Section 3.4 and Algorithm 2 of the paper.

Copies of training embeddings, shifted by the group mean differences inside their class,
are added to the cells that the minimal plan asks for; the embeddings themselves stay as they are.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .stats import cell_stats


def minimal_plan(counts):
    """Fewest additions that make class and group independent: integer additions (C, G), target group shares q (G,).

    Class y needs m_y = max_z n_zy / q_z samples and cell (y, z) gets a_zy = m_y q_z - n_zy; sum_y m_y is convex in q.
    Two groups: bounded 1-D search plus the kinks q_1 = P(z = 1 | y).  More groups: Nelder-Mead in log-ratio
    coordinates, started at the overall group shares and at the shares of every class.
    Requires every cell count to be positive.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape[1] == 2:
        def total(q1):
            return np.maximum(counts[:, 1] / q1, counts[:, 0] / (1 - q1)).sum()

        res = minimize_scalar(total, bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-12})
        cand = np.r_[res.x, counts[:, 1] / counts.sum(1)]
        q1 = cand[np.argmin([total(c) for c in cand])]
        q = np.array([1 - q1, q1])
    else:
        def shares(v):
            e = np.exp(np.r_[0.0, v])
            return e / e.sum()

        def total(v):
            return (counts / shares(v)).max(1).sum()

        starts = np.vstack([counts.sum(0), counts])
        fits = [minimize(total, np.log(s[1:] / s[0]), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
                for s in starts]
        q = shares(min(fits, key=lambda r: r.fun).x)
    plan = (counts / q).max(1)[:, None] * q - counts
    return np.round(plan).clip(0).astype(int), q


def plan_sources(y, z, plan, rng):
    """Source rows and target groups of the additions plan[y, z]: rows of the other groups of class y.

    Distinct sources first; rows repeat only when a cell needs more copies than there are sources.
    """
    src, target = [np.zeros(0, dtype=int)], [np.zeros(0, dtype=int)]
    for c, g in zip(*np.nonzero(plan)):
        pool = np.flatnonzero((y == c) & (z != g))
        k = int(plan[c, g])
        src.append(rng.choice(pool, k, replace=False) if k <= len(pool) else np.r_[pool, rng.choice(pool, k - len(pool))])
        target.append(np.full(k, g))
    return np.concatenate(src), np.concatenate(target)


def mean_shift(X, y, z, src, target, means):
    """Copies of the rows src moved to the groups `target`: x + mu_{target, y} - mu_{z, y}.

    Copies made from cell (y, z') get the mean of cell (y, target) and keep the scatter of their source cell.
    means: cell means (C, G, d) of the training set.
    """
    return X[src] + (means[y[src], target] - means[y[src], z[src]]).astype(X.dtype)


def bma(X, y, z, rng=None, n_classes=None, n_groups=None):
    """Bias-Matrix Augmentation (Algorithm 2): the copies (X_new, y_new, z_new) to append to the training set.

    Only the model is trained on the augmented set; test embeddings are not changed.
    """
    rng = np.random.default_rng(rng)
    means, counts = cell_stats(X, y, z, n_classes, n_groups)
    plan, _ = minimal_plan(counts)
    src, target = plan_sources(y, z, plan, rng)
    return mean_shift(X, y, z, src, target, means), y[src], target
