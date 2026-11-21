"""Bias-Matrix Editing (BME) and Bias-Matrix Augmentation (BMA).

    eraser = bme(X_train, y_train, z_train)          # edit embeddings (Algorithm 1)
    X_new, y_new, z_new = bma(X_train, y_train, z_train)   # copies for the training set (Algorithm 2)
"""
from .bma import bma, mean_shift, minimal_plan, plan_sources
from .bme import (Eraser, bme, bme_eraser, conditional_bias_matrix, numerical_rank, permutation_null, select_rank,
                  sequential_rank, shuffle_within)
from .stats import cell_stats, count_table, covariance, mutual_information, sym_power, table_shape

__all__ = [
    "bme", "bme_eraser", "Eraser", "conditional_bias_matrix", "numerical_rank",
    "select_rank", "sequential_rank", "permutation_null", "shuffle_within",
    "bma", "minimal_plan", "plan_sources", "mean_shift",
    "cell_stats", "count_table", "covariance", "sym_power", "mutual_information", "table_shape",
]
