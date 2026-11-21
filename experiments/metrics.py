"""Recall-gap metrics of test predictions: TPR by (class, group) cell, RMS gap, bootstrap-corrected gap, group recalls.

GAP_y = max_z TPR_zy - min_z TPR_zy,  GAP^RMS = sqrt(mean_y GAP_y^2)  (Section 3.1 of the paper).
"""
import numpy as np


def tpr_by_cell(y, pred, z, n_classes, n_groups=2):
    """Recall of each (class, group) cell: (C, G); 0 for empty cells."""
    cell = y * n_groups + z
    hit = np.bincount(cell, weights=(pred == y), minlength=n_classes * n_groups)
    cnt = np.bincount(cell, minlength=n_classes * n_groups)
    return (hit / np.maximum(cnt, 1)).reshape(n_classes, n_groups)


def gap_sq(tpr):
    """Mean over classes of the squared gap between the best and the worst group."""
    return float(((tpr.max(1) - tpr.min(1)) ** 2).mean())


def _rms(v):
    return float(np.sqrt(max(v, 0.0)))


def corrected_gap(y, pred, z, n_classes, n_groups=2, n_boot=500, seed=0):
    """Bootstrap-corrected RMS gap and its 95% interval (test resampling, model fixed).

    The plug-in RMS of noisy per-class gaps is biased upward: bias correction and basic interval
    on the mean-squared-gap scale (a percentile interval would add the bias a second time).
    """
    rng = np.random.default_rng(seed)
    theta = gap_sq(tpr_by_cell(y, pred, z, n_classes, n_groups))
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boot.append(gap_sq(tpr_by_cell(y[i], pred[i], z[i], n_classes, n_groups)))
    q_lo, q_hi = np.quantile(boot, [0.025, 0.975])
    return _rms(2 * theta - np.mean(boot)), [_rms(2 * theta - q_hi), _rms(2 * theta - q_lo)]


def group_roles(counts):
    """Smallest and largest group of each class in the training table: (minority, majority) index arrays.

    Ties: first smallest, last largest (two equal groups still get different roles).
    """
    return counts.argmin(1), counts.shape[1] - 1 - counts[:, ::-1].argmax(1)


def score(y, pred, z, n_classes, n_groups, minority, majority):
    """Test metrics of one prediction set.

    acc; gap_rms (plug-in) and gap_rms_bc with gap_rms_ci (bootstrap-corrected, 95%);
    tpr_minority / tpr_majority: recall of the smallest / largest training group of each class, averaged over classes.
    """
    tpr = tpr_by_cell(y, pred, z, n_classes, n_groups)
    rows = np.arange(n_classes)
    gap_bc, gap_ci = corrected_gap(y, pred, z, n_classes, n_groups)
    return {"acc": float((pred == y).mean()), "gap_rms": float(np.sqrt(gap_sq(tpr))), "gap_rms_bc": gap_bc, "gap_rms_ci": gap_ci,
            "tpr_minority": float(tpr[rows, minority].mean()), "tpr_majority": float(tpr[rows, majority].mean())}
