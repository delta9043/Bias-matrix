"""Numerical checks of the propositions and identities behind BME and BMA (synthetic data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bias_matrix as bm  # noqa: E402

C, DIM, N = 6, 40, 6000


@pytest.fixture(scope="module")
def data():
    """Gaussian data with class effects, a shared group direction and class-specific group effects (two groups)."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, C, N)
    z = (rng.random(N) < np.linspace(0.15, 0.85, C)[y]).astype(int)  # P(z=1|y) varies with y
    mix = rng.standard_normal((DIM, DIM)) / np.sqrt(DIM) + np.diag(np.linspace(0.5, 3, DIM))
    X = rng.standard_normal((N, DIM)) @ mix
    X += rng.standard_normal((C, DIM))[y] * 2
    X += np.outer(z, rng.standard_normal(DIM))
    X += (rng.standard_normal((C, DIM)) * 0.5)[y] * z[:, None]
    return X, y, z


@pytest.fixture(scope="module")
def data3():
    """Three groups with class-dependent shares and cell-specific mean shifts."""
    rng = np.random.default_rng(1)
    G, n = 3, 5000
    y = rng.integers(0, C, n)
    probs = rng.dirichlet(np.full(G, 2.0), C)
    z = np.array([rng.choice(G, p=probs[c]) for c in y])
    X = rng.standard_normal((n, DIM)) @ (rng.standard_normal((DIM, DIM)) / np.sqrt(DIM) + np.eye(DIM))
    X += rng.standard_normal((C, DIM))[y] * 2 + rng.standard_normal((C, G, DIM))[y, z]
    return X, y, z, G


# ------------------------------------------------------------ bias matrix
def test_bias_matrix_is_within_class_group_scatter(data):
    """Section 3.2: B B^T = (1/n) sum_{y,z} n_zy (mu_zy - mu_y)(mu_zy - mu_y)^T; rank C for two groups."""
    X, y, z = data
    B = bm.conditional_bias_matrix(X, y, z)
    means, counts = bm.cell_stats(X, y, z)
    class_mean = (means * counts[..., None]).sum(1) / counts.sum(1, keepdims=True)
    dev = means - class_mean[:, None]
    assert B.shape == (DIM, 2 * C)
    assert np.allclose(B @ B.T, np.einsum("cg,cgi,cgj->ij", counts, dev, dev) / N, atol=1e-10)
    assert bm.numerical_rank(np.linalg.svd(B, compute_uv=False)) == C


def test_general_forms_agree_with_binary_formulas(data):
    """Two groups: the (d, C G) matrix has the B B^T of the columns sqrt(w_y / n) (mu_1y - mu_0y), w_y = n_0y n_1y / n_y;
    the same holds for the cross-covariance with the standardized within-class-centered z; the uniform variant spans the
    plain differences; the general minimal plan equals the closed 1-D form."""
    X, y, z = data
    means, counts = bm.cell_stats(X, y, z)
    gaps = means[:, 1] - means[:, 0]
    w = counts[:, 0] * counts[:, 1] / counts.sum(1)
    B2 = (np.sqrt(w / N)[:, None] * gaps).T
    B = bm.conditional_bias_matrix(X, y, z)
    assert np.allclose(B @ B.T, B2 @ B2.T, atol=1e-12)
    p_y = np.array([z[y == c].mean() for c in range(C)])
    Zt = np.zeros((N, C))
    Zt[np.arange(N), y] = z - p_y[y]
    Zt /= Zt.std(0)
    assert np.allclose(B2, (X - X.mean(0)).T @ Zt / N, atol=1e-10)
    Bu = bm.conditional_bias_matrix(X, y, z, weights="uniform")
    assert np.allclose(Bu @ Bu.T, gaps.T @ gaps / 2, atol=1e-12)
    # Minimal Plan: class y needs max(n_1y / q, n_0y / (1 - q)) rows; total minimal over a grid of q
    plan, q = bm.minimal_plan(counts)
    m = np.maximum(counts[:, 1] / q[1], counts[:, 0] / q[0])
    assert np.array_equal(plan, np.round(np.stack([m * q[0], m * q[1]], 1) - counts).astype(int))
    grid = np.linspace(0.01, 0.99, 9801)
    totals = [np.maximum(counts[:, 1] / g, counts[:, 0] / (1 - g)).sum() for g in grid]
    assert m.sum() <= min(totals) + 1e-6
    assert bm.mutual_information(counts + plan) < 1e-5 < bm.mutual_information(counts)  # integer plan: independent up to rounding
    assert plan.sum() <= np.abs(counts[:, 1] - counts[:, 0]).sum() + C  # q = 1/2 costs sum |n_1y - n_0y|


# ------------------------------------------------------------------- BME
def test_projection_is_minimal_change():
    """(I - U U^T) x is the closest point with U^T x' = 0 (centered coordinates)."""
    rng = np.random.default_rng(2)
    U, _ = np.linalg.qr(rng.standard_normal((DIM, 4)))
    x = rng.standard_normal(DIM)
    x_proj = x - U @ (U.T @ x)
    assert np.allclose(U.T @ x_proj, 0)
    assert np.isclose(np.linalg.norm(x - x_proj) ** 2, np.linalg.norm(U.T @ x) ** 2)
    for _ in range(200):  # any other feasible point is farther
        v = rng.standard_normal(DIM)
        x_other = x_proj + (v - U @ (U.T @ v))
        assert np.linalg.norm(x - x_other) >= np.linalg.norm(x - x_proj)


def test_bme_edits_bias_spectrum(data):
    """Prop 1: after the edit B' = (I - U_k U_k^T) B, top-k singular values 0 and the rest unchanged; soft: s_j -> lambda_j s_j."""
    X, y, z = data
    B = bm.conditional_bias_matrix(X, y, z)
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    mean = X.mean(0)
    hard = bm.bme_eraser(B, mean, k=2)
    B_new = bm.conditional_bias_matrix(hard(X), y, z)
    assert np.allclose(B_new, (np.eye(DIM) - U[:, :2] @ U[:, :2].T) @ B, atol=1e-10)
    assert np.allclose(B_new, (U[:, 2:] * s[2:]) @ Vt[2:], atol=1e-10)
    assert np.allclose(np.linalg.svd(B_new, compute_uv=False)[: C - 2], s[2:C], atol=1e-10)
    lam = np.array([0.0, 0.3, 0.8])
    s_new = np.linalg.svd(bm.conditional_bias_matrix(bm.bme_eraser(B, mean, k=3, shrink=lam)(X), y, z), compute_uv=False)
    assert np.allclose(np.sort(s_new[:C]), np.sort(np.r_[lam * s[:3], s[3:C]]), atol=1e-10)
    # Eckart-Young: top-k directions leave the least residual bias energy
    resid = np.linalg.norm(B_new) ** 2
    assert np.isclose(resid, (s[2:] ** 2).sum())
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((DIM, 2)))
    assert np.linalg.norm(bm.conditional_bias_matrix(bm.Eraser(Q, Q, mean)(X), y, z)) ** 2 >= resid


def test_bme_function_matches_eraser_and_k_is_capped(data):
    """bme() = bme_eraser() on the statistics of the training set; k above the rank of B removes the same space."""
    X, y, z = data
    B = bm.conditional_bias_matrix(X, y, z)
    mean, cov = bm.covariance(X)
    assert np.allclose(bm.bme(X, y, z).matrix(), bm.bme_eraser(B, mean, cov).matrix())
    assert np.allclose(bm.bme(X, y, z, k=2, oblique=False).matrix(), bm.bme_eraser(B, mean, k=2).matrix())
    assert np.allclose(bm.bme_eraser(B, mean, k=2 * C).matrix(), bm.bme_eraser(B, mean).matrix())
    with pytest.raises(ValueError):
        bm.conditional_bias_matrix(X, y, z, weights="other")


@pytest.mark.parametrize("oblique", [False, True])
def test_full_rank_bme_equalizes_group_means(data, oblique):
    """Prop 2: k = rank(B) -> mu'_zy = mu'_y in every class -> equal mean output of any linear model across groups."""
    X, y, z = data
    Xe = bm.bme(X, y, z, oblique=oblique)(X)
    means, _ = bm.cell_stats(Xe, y, z)
    assert np.allclose(means[:, 1], means[:, 0], atol=1e-9)
    W = np.random.default_rng(5).standard_normal((DIM, C))
    assert np.allclose(means[:, 1] @ W, means[:, 0] @ W, atol=1e-8)
    # Class-Aware Logistic Group Probe at w = 0: gradient sum_i (z_i - P(z=1|y_i)) x_i = 0
    p_y = np.array([z[y == c].mean() for c in range(C)])
    assert np.allclose((z - p_y[y]) @ Xe / N, 0, atol=1e-9)
    # The Class-Free Matrix Alone Does Not Equalize the Means inside the Classes
    Bg = bm.conditional_bias_matrix(X, np.zeros(N, dtype=int), z)
    mg, _ = bm.cell_stats(bm.bme_eraser(Bg, X.mean(0))(X), y, z)
    assert np.abs(mg[:, 1] - mg[:, 0]).max() > 0.1


def test_orthogonal_and_oblique_edits_give_same_retrained_predictions(data):
    """Erasers with the same null space span(B) leave the same linear function class:
    least-squares models refit on orthogonally vs. obliquely edited features predict identically."""
    X, y, z = data
    mean, cov = bm.covariance(X)
    B = bm.conditional_bias_matrix(X, y, z)
    t = np.eye(C)[y] - np.eye(C)[y].mean(0)
    preds = []
    for er in (bm.bme_eraser(B, mean), bm.bme_eraser(B, mean, cov)):
        E = er(X) - mean
        preds.append(E @ np.linalg.lstsq(E, t, rcond=1e-10)[0])
        assert np.allclose(er.matrix() @ B, 0, atol=1e-9)  # span(B) is annihilated
    assert np.allclose(preds[0], preds[1], atol=1e-7)


def test_soft_edit_is_invertible_and_equals_ridge_penalty(data):
    """Soft editing keeps every linear function (w' = A^{-T} w); ridge on edited data = direction-wise ridge on the original."""
    X, y, z = data
    mean = X.mean(0)
    B = bm.conditional_bias_matrix(X, y, z)
    er = bm.bme_eraser(B, mean, k=3, shrink=np.array([0.2, 0.5, 0.9]))
    Amat = er.matrix()
    w = np.random.default_rng(4).standard_normal(DIM)
    Xc = X - mean
    assert np.allclose(Xc @ w, (er(X) - mean) @ np.linalg.solve(Amat.T, w))
    rho, t = 5.0, z - z.mean()
    Xs = er(X) - mean
    w_shrunk = np.linalg.solve(Xs.T @ Xs + rho * np.eye(DIM), Xs.T @ t)
    Ainv = np.linalg.inv(Amat)
    w_gen = np.linalg.solve(Xc.T @ Xc + rho * Ainv @ Ainv, Xc.T @ t)
    assert np.allclose(Amat @ w_shrunk, w_gen)  # same predictor on the original features


def test_oblique_edit_scales_whitened_spectrum(data):
    """Oblique edit: the singular values of cov^{-1/2} B (canonical correlations) are multiplied by lambda on the top-k directions."""
    X, y, z = data
    mean, cov = bm.covariance(X)
    B = bm.conditional_bias_matrix(X, y, z)
    W = bm.sym_power(cov, -0.5)
    c = np.linalg.svd(W @ B, compute_uv=False)[:C]
    lam = np.array([0.2, 0.5, 0.9])
    Amat = bm.bme_eraser(B, mean, cov, k=3, shrink=lam).matrix()
    assert np.allclose(np.sort(np.linalg.svd(W @ Amat @ B, compute_uv=False)[:C]), np.sort(np.r_[lam * c[:3], c[3:]]), atol=1e-10)
    assert c.max() < 1
    hard = bm.bme_eraser(B, mean, cov, k=2).matrix()
    assert np.allclose(np.sort(np.linalg.svd(W @ hard @ B, compute_uv=False)[:C])[2:], np.sort(c[2:]), atol=1e-10)
    assert not np.allclose(Amat, Amat.T)  # oblique: not an orthogonal projection


def test_class_free_oblique_edit_matches_leace(data):
    """LEACE baseline (class-free bias matrix, oblique): zero cross-covariance with z, agreement with concept-erasure,
    smaller change than the orthogonal projection."""
    torch = pytest.importorskip("torch")
    ce = pytest.importorskip("concept_erasure")
    X, y, z = data
    mean, cov = bm.covariance(X)
    B = bm.conditional_bias_matrix(X, np.zeros(N, dtype=int), z)
    Xe = bm.bme_eraser(B, mean, cov)(X)
    assert np.allclose((Xe - Xe.mean(0)).T @ (z - z.mean()) / N, 0, atol=1e-10)
    fitter = ce.LeaceFitter(DIM, 1, dtype=torch.float64, shrinkage=False, constrain_cov_trace=False, svd_tol=1e-9)
    fitter.update(torch.tensor(X), torch.tensor(z[:, None], dtype=torch.float64))
    assert np.allclose(Xe, fitter.eraser(torch.tensor(X)).numpy(), atol=1e-8)
    proj = bm.bme_eraser(B, mean)(X)
    assert np.allclose((proj - proj.mean(0)).T @ (z - z.mean()) / N, 0, atol=1e-10)
    assert np.linalg.norm(Xe - X) <= np.linalg.norm(proj - X) + 1e-9  # least-squares optimality


def test_sequential_rank_recovers_planted_rank():
    """Sequential permutation test finds a planted rank-2 conditional bias; no signal -> k = 0."""
    rng = np.random.default_rng(9)
    n, d, c = 20000, 60, 8
    y = rng.integers(0, c, n)
    z = (rng.random(n) < np.linspace(0.3, 0.7, c)[y]).astype(int)
    noise = rng.standard_normal((n, d)) * np.linspace(0.5, 2, d) + rng.standard_normal((c, d))[y]
    dirs = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    coef = np.stack([np.ones(c), np.linspace(-1, 1, c)], 1) * [1.0, 0.6]  # shared + class-varying group effect
    X = noise + z[:, None] * (coef[y] @ dirs.T)
    for data_x, expected in [(X, 2), (noise, 0)]:
        k, sv, thr = bm.select_rank(data_x, y, z, n_perm=100, rng=rng)
        assert k == expected and len(sv) == len(thr) == c
    # Per-Index Comparison Over-Selects after a Strong Component
    s = np.linalg.svd(bm.conditional_bias_matrix(X, y, z), compute_uv=False)[:c]
    s_null = np.linalg.svd(bm.permutation_null(X, y, z, 100, rng), compute_uv=False)[:, :c]
    assert (s > np.quantile(s_null, 0.95, axis=0)).sum() > 2


# ------------------------------------------------------------------- BMA
def test_mean_shift_copies_and_squared_loss_identity(data):
    """Prop 3: copies of cell (y, z') moved to group z have mean mu_zy and the covariance of their source cell;
    l(x) + l(x~) = 2 l(c) + (w^T (mu_zy - mu_z'y))^2 / 2 with the midpoint c; summed over a set of sources the
    extra term is (n / 2) ||B_a^T w||^2, a ridge penalty along the columns of the bias matrix."""
    X, y, z = data
    means, counts = bm.cell_stats(X, y, z)
    everyone = np.arange(N)
    Xcf = bm.mean_shift(X, y, z, everyone, 1 - z, means)
    m_cf, _ = bm.cell_stats(Xcf, y, z)
    assert np.allclose(m_cf[:, 0], means[:, 1]) and np.allclose(m_cf[:, 1], means[:, 0])
    c0 = (y == 2) & (z == 0)
    assert np.allclose(np.cov(Xcf[c0].T), np.cov(X[c0].T))
    # Per-Sample Identity
    rng = np.random.default_rng(13)
    w, b0, t = rng.standard_normal(DIM), 0.3, rng.standard_normal(N)
    loss = lambda A: (A @ w + b0 - t) ** 2  # noqa: E731
    gaps = means[:, 1] - means[:, 0]
    mid = (X + Xcf) / 2
    assert np.allclose(loss(X) + loss(Xcf), 2 * loss(mid) + 0.5 * (gaps[y] @ w) ** 2)
    # Midpoints of All Pairs: no group difference left inside the classes
    assert np.allclose(bm.conditional_bias_matrix(mid, y, z), 0, atol=1e-10)
    # Any Set of Sources S with a_y Copies in Class y: penalty = (N / 2) ||B_a^T w||^2, B_a = [sqrt(a_y / n) (mu_1y - mu_0y)]
    S = rng.random(N) < np.where(z == 1, 0.1, 0.5)
    a = np.bincount(y[S], minlength=C)
    penalty = (loss(X).sum() + loss(Xcf)[S].sum()) - (loss(X)[~S].sum() + 2 * loss(mid)[S].sum())
    B_a = (np.sqrt(a / N)[:, None] * gaps).T
    assert np.isclose(penalty, 0.5 * N * np.linalg.norm(B_a.T @ w) ** 2)


def test_bma_makes_class_and_group_independent(data):
    """Algorithm 2: the copies fill the cells of the minimal plan; the augmented table is independent (up to rounding)."""
    X, y, z = data
    counts = bm.count_table(y, z)
    plan, q = bm.minimal_plan(counts)
    X_new, y_new, z_new = bm.bma(X, y, z, rng=0)
    assert np.array_equal(bm.count_table(y_new, z_new, C, 2), plan)
    after = bm.count_table(np.r_[y, y_new], np.r_[z, z_new])
    assert bm.mutual_information(after) < 1e-5 and np.allclose(after / after.sum(1, keepdims=True), q, atol=2e-3)
    # Sources Come from the Other Group of the Same Class; copies sit at the mean of their new cell
    src, target = bm.plan_sources(y, z, plan, np.random.default_rng(0))
    assert np.array_equal(y[src], y_new) and (z[src] != target).all() and np.array_equal(target, z_new)
    means, _ = bm.cell_stats(X, y, z)
    c, g = np.unravel_index(plan.argmax(), plan.shape)  # cell with the most additions
    cell = (y_new == c) & (z_new == g)
    spread = X[(y == c) & (z != g)].std(0).max()
    assert np.allclose(X_new[cell].mean(0), means[c, g], atol=4 * spread / np.sqrt(cell.sum()))
    assert bm.plan_sources(y, z, np.zeros_like(plan), np.random.default_rng(0))[0].size == 0  # nothing to add


def test_three_groups(data3):
    """Forms for G groups: B B^T is the within-class between-group scatter with rank C (G - 1), editing k directions
    leaves the other singular values, a full hard edit equalizes the group means within classes (both projections),
    the minimal plan makes class and group independent with the fewest additions, mean-shift copies keep the covariance."""
    X, y, z, G = data3
    n = len(X)
    means, counts = bm.cell_stats(X, y, z)
    B = bm.conditional_bias_matrix(X, y, z)
    rank = C * (G - 1)

    # Scatter Identity and Rank
    mu_y = (counts[:, :, None] * means).sum(1) / counts.sum(1)[:, None]
    scatter = sum(counts[c, g] * np.outer(means[c, g] - mu_y[c], means[c, g] - mu_y[c]) for c in range(C) for g in range(G)) / n
    assert np.allclose(B @ B.T, scatter)
    s = np.linalg.svd(B, compute_uv=False)
    assert bm.numerical_rank(s) == rank

    # Editing k Directions: remaining singular values unchanged
    mean, cov = bm.covariance(X)
    k = 4
    B_k = bm.conditional_bias_matrix(bm.bme_eraser(B, mean, k=k)(X), y, z)
    assert np.allclose(np.linalg.svd(B_k, compute_uv=False)[: rank - k], s[k:rank])

    # Full Hard Edit: group means agree within every class
    for eraser in (bm.bme_eraser(B, mean), bm.bme_eraser(B, mean, cov)):
        mu_e, _ = bm.cell_stats(eraser(X), y, z)
        assert np.abs(mu_e - mu_e.mean(1, keepdims=True)).max() < 1e-8

    # Minimal Plan: independence after the additions, total no larger than on a grid over the simplex
    plan, q = bm.minimal_plan(counts)
    assert bm.mutual_information(counts + plan) < 1e-5
    total = lambda v: (counts / v).max(1).sum()  # noqa: E731
    grid = [np.array([a, b, 1 - a - b]) for a in np.linspace(0.02, 0.96, 95) for b in np.linspace(0.02, 0.96, 95) if a + b < 0.98]
    assert total(q) <= min(total(v) for v in grid) + 1e-6

    # Mean-Shift Copies: target cell mean, source cell covariance; BMA fills the plan from the other groups
    src = np.flatnonzero((y == 0) & (z == 1))
    copies = bm.mean_shift(X, y, z, src, np.full(len(src), 2), means)
    assert np.allclose(copies.mean(0), means[0, 2]) and np.allclose(np.cov(copies.T), np.cov(X[src].T))
    X_new, y_new, z_new = bm.bma(X, y, z, rng=0)
    assert np.array_equal(bm.count_table(y_new, z_new, C, G), plan)


# --------------------------------------------------------------- metrics
def test_recall_gap_metrics_and_bootstrap_correction():
    """Recall-gap bookkeeping on a hand example; bootstrap correction when the true gap is zero; three groups: best minus worst."""
    from experiments import metrics as M

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    z = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    pred = np.array([0, 1, 0, 0, 1, 1, 0, 0])
    tpr = M.tpr_by_cell(y, pred, z, 2)
    assert np.allclose(tpr, [[0.5, 1.0], [1.0, 0.0]])
    assert np.isclose(M.gap_sq(tpr), (0.25 + 1) / 2)
    # Roles from the Training Table: minority = group 1 in class 0, group 0 in class 1
    minority, majority = M.group_roles(np.array([[30.0, 10.0], [5.0, 20.0]]))
    assert np.array_equal(minority, [1, 0]) and np.array_equal(majority, [0, 1])
    assert np.array_equal(M.group_roles(np.array([[7.0, 7.0]]))[1], [1])  # tie: different roles
    r = M.score(y, pred, z, 2, 2, minority, majority)
    assert np.isclose(r["acc"], 5 / 8) and np.isclose(r["gap_rms"], np.sqrt((0.25 + 1) / 2))
    assert np.isclose(r["tpr_minority"], 1.0) and np.isclose(r["tpr_majority"], 0.25)
    assert np.isclose(M.gap_sq(np.array([[0.9, 0.5, 0.7], [0.2, 0.2, 0.6]])), (0.16 + 0.16) / 2)
    # Zero True Gap: the plug-in RMS is pure noise, the corrected estimate is far smaller and the interval starts at 0
    rng = np.random.default_rng(11)
    n, c = 20000, 10
    y = rng.integers(0, c, n)
    z = (rng.random(n) < 0.3).astype(int)
    pred = np.where(rng.random(n) < 0.7, y, (y + 1) % c)  # same recall for both groups
    plug = np.sqrt(M.gap_sq(M.tpr_by_cell(y, pred, z, c)))
    gap_bc, ci = M.corrected_gap(y, pred, z, c, n_boot=300)
    assert gap_bc < 0.75 * plug and ci[0] == 0.0 and ci[1] < 2 * plug
