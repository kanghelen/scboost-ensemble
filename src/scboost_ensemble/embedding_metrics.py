"""Pairwise similarity metrics between the embeddings in ``adata.obsm``.

scBoost stacks base classifiers trained on several unsupervised embeddings of the
same cells. Stacking can only add accuracy where the embeddings carry different
information, so this module measures, without labels, how redundant two of them are.

Functions
---------
linear_cka               — linear centered kernel alignment (Kornblith et al., ICML 2019).
                           Symmetric, in [0, 1], invariant to rotation, isotropic scaling
                           and translation; defined for embeddings of different width.
linear_fit               — least-squares map X -> Y after standardizing both: R² (fraction
                           of Y's variance explained), per-cell residual norms, rank of X.
                           Asymmetric by design (a narrow embedding is easier to predict).
paired_distance          — raw per-cell L2 distance between two standardized embeddings.
                           Equal width only, and meaningful only when the two coordinate
                           systems are aligned (one method under two settings). Between
                           different methods the axes are arbitrary; use ``linear_fit``.
compare_embeddings       — all of the above for every ordered pair of keys, as one table.
similarity_matrix        — pivot one metric of that table into a square DataFrame.
plot_similarity_heatmap  — heatmap of such a matrix; returns the Axes.

Original formulations, kept for comparison (see that section for how they differ)
lstsq_measure            — least-squares map on mean-norm-scaled but uncentered embeddings.
iterative_mapping        — gradient descent on the mean per-cell residual norm (needs torch).
iterative_map_test       — iterative_mapping for every ordered pair, as one table.
compare_formulations     — the new metrics and both originals on the same cells, merged.

Scale conventions
-----------------
``standardize`` centers every column and rescales so the mean per-cell norm is 1. Residual
and distance columns are therefore in units of "a typical cell's distance from the
centroid", comparable across pairs and datasets. R² and CKA are unitless.

Command line
------------
python -m scboost_ensemble.embedding_metrics in.h5ad --obsm_keys Python_PCA scVI NMF \\
    --out pairs.tsv --heatmap cka.png
"""

from __future__ import annotations
import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

METRICS = ("cka", "r2", "residual_mean", "residual_median", "residual_max",
           "raw_distance_mean", "raw_distance_median", "raw_distance_max")
_SELF_VALUE = {m: (1.0 if m in ("cka", "r2") else 0.0) for m in METRICS}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _dense_float64(X, name: str = "embedding") -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name}: expected a 2-d (n_cells, d) array, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: contains NaN or inf")
    return X


def _check_rows(X: np.ndarray, Y: np.ndarray) -> None:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"embeddings must describe the same cells: {X.shape[0]} vs {Y.shape[0]} rows")


def center(X, name: str = "embedding") -> np.ndarray:
    """Return a float64 copy with every column centered at 0."""
    X = _dense_float64(X, name)
    return X - X.mean(axis=0, keepdims=True)


def standardize(X, name: str = "embedding") -> np.ndarray:
    """Center the columns and rescale so the mean per-cell L2 norm is 1 (float64 copy).

    A constant embedding (all cells identical) is returned centered but unscaled."""
    X = center(X, name)
    scale = np.linalg.norm(X, axis=1).mean()
    return X / scale if scale > 0 else X


# ---------------------------------------------------------------------------
# Pairwise metrics
# ---------------------------------------------------------------------------

def _cka_centered(Xc: np.ndarray, Yc: np.ndarray) -> float:
    cross = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    nx, ny = np.linalg.norm(Xc.T @ Xc, "fro"), np.linalg.norm(Yc.T @ Yc, "fro")
    if nx == 0 or ny == 0:
        return float("nan")
    return float(cross / (nx * ny))


def linear_cka(X, Y) -> float:
    """Linear centered kernel alignment between two embeddings of the same cells.

    CKA(X, Y) = ||Xcᵀ Yc||²_F / (||Xcᵀ Xc||_F ||Ycᵀ Yc||_F), with Xc, Yc column-centered,
    the linear form in Kornblith et al. (2019). Symmetric, in [0, 1]; 1 means one embedding
    is a rotation + scaling + shift of the other. The widths may differ. Cost is
    O(n (d_x + d_y)²), so 100k cells × 100 dims takes well under a second.

    Parameters
    ----------
    X, Y : array-like, shape (n_cells, d_x) and (n_cells, d_y)

    Returns
    -------
    float — NaN if either embedding is constant."""
    Xc, Yc = center(X, "X"), center(Y, "Y")
    _check_rows(Xc, Yc)
    return _cka_centered(Xc, Yc)


def _fit_standardized(Xs: np.ndarray, Ys: np.ndarray) -> dict:
    coef, _, rank, _ = np.linalg.lstsq(Xs, Ys, rcond=None)
    resid = Ys - Xs @ coef
    norms = np.linalg.norm(resid, axis=1)
    ss_tot = float((Ys ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2,
            "residual_mean": float(norms.mean()),
            "residual_median": float(np.median(norms)),
            "residual_max": float(norms.max()),
            "rank": int(rank),
            "coef": coef,
            "residual_norms": norms}


def linear_fit(X, Y) -> dict:
    """Best linear map from X to Y (least squares, closed form) and how well it does.

    Both embeddings are standardized first, so no intercept is needed and the residuals
    are in units of a typical cell's distance from Y's centroid.

    Parameters
    ----------
    X : array-like, shape (n_cells, d_x) — predictor embedding
    Y : array-like, shape (n_cells, d_y) — target embedding

    Returns
    -------
    dict with keys
        r2              : 1 − ||Y − XW||² / ||Y||², fraction of Y's variance explained
        residual_mean, residual_median, residual_max : per-cell residual norm summaries
        rank            : numerical rank of the standardized X
        coef            : ndarray (d_x, d_y), the fitted map W
        residual_norms  : ndarray (n_cells,), per-cell ||y − xW||"""
    Xs, Ys = standardize(X, "X"), standardize(Y, "Y")
    _check_rows(Xs, Ys)
    return _fit_standardized(Xs, Ys)


def paired_distance(X, Y) -> np.ndarray:
    """Per-cell L2 distance between two standardized embeddings of equal width.

    This compares coordinates directly, so it is only meaningful when both embeddings
    share an axis system (e.g. the same method run twice). Between different methods the
    axes are arbitrary and this number mostly measures the rotation between them; use
    ``linear_fit`` or ``linear_cka`` instead.

    Returns
    -------
    ndarray (n_cells,)"""
    Xs, Ys = standardize(X, "X"), standardize(Y, "Y")
    _check_rows(Xs, Ys)
    if Xs.shape[1] != Ys.shape[1]:
        raise ValueError(f"paired_distance needs equal width, got {Xs.shape[1]} and {Ys.shape[1]}")
    return np.linalg.norm(Xs - Ys, axis=1)


# ---------------------------------------------------------------------------
# All pairs
# ---------------------------------------------------------------------------

def _prepare(embeddings, keys: Optional[Sequence[str]], max_cells: Optional[int], random_state: int):
    """Resolve keys, load the matrices as float64 and subsample the same cells for every key.

    Returns (keys, {key: ndarray}, n_cells)."""
    source = embeddings.obsm if hasattr(embeddings, "obsm") else embeddings
    keys = list(source.keys()) if keys is None else list(keys)
    missing = [k for k in keys if k not in source]
    if missing:
        raise KeyError(f"embedding keys not found: {missing}; available: {list(source.keys())}")
    if len(keys) < 2:
        raise ValueError("need at least two embeddings to compare")
    mats = {k: _dense_float64(source[k], k) for k in keys}
    n_rows = {m.shape[0] for m in mats.values()}
    if len(n_rows) != 1:
        raise ValueError(f"all embeddings must have the same number of cells, got {sorted(n_rows)}")
    n_cells = n_rows.pop()
    if max_cells is not None and n_cells > max_cells:
        idx = np.sort(np.random.default_rng(random_state).choice(n_cells, max_cells, replace=False))
        mats = {k: m[idx] for k, m in mats.items()}
        n_cells = max_cells
    return keys, mats, n_cells


def _ordered_pairs(keys: Sequence[str]) -> list:
    return [(a, b) for a in keys for b in keys if a != b]


def compare_embeddings(embeddings, keys: Optional[Sequence[str]] = None,
                       max_cells: Optional[int] = None, random_state: int = 0,
                       verbose: bool = True) -> pd.DataFrame:
    """Every pairwise metric for every ordered pair of embeddings.

    Parameters
    ----------
    embeddings : AnnData or Mapping[str, array]
        Embeddings of the same cells. For an AnnData, ``.obsm`` is used.
    keys : sequence of str, optional
        Which entries to compare; default all. For the scBoost benchmark h5ads pass the
        seven single-method keys explicitly, since the ``scBoost_concat_*`` entries are
        200–700-d and only restate their parts.
    max_cells : int, optional
        Subsample this many cells (the same cells for every key) before computing.
    random_state : int
        Seed for the subsample.
    verbose : bool
        Print one line per pair.

    Returns
    -------
    DataFrame, one row per ordered pair (``from`` ≠ ``to``) with columns
        from, to, n_cells, dim_from, dim_to, cka, r2, residual_mean, residual_median,
        residual_max, rank_from, raw_distance_mean, raw_distance_median, raw_distance_max.
    ``cka`` is the same in both directions; ``r2`` and the residuals are for the map
    from → to. The ``raw_distance_*`` columns are NaN when the widths differ."""
    keys, mats, n_cells = _prepare(embeddings, keys, max_cells, random_state)
    std = {k: standardize(m, k) for k, m in mats.items()}
    del mats

    pairs = _ordered_pairs(keys)
    rows = []
    for i, (a, b) in enumerate(pairs, 1):
        Xs, Ys = std[a], std[b]
        fit = _fit_standardized(Xs, Ys)
        same_width = Xs.shape[1] == Ys.shape[1]
        dist = np.linalg.norm(Xs - Ys, axis=1) if same_width else None
        row = {"from": a, "to": b, "n_cells": n_cells, "dim_from": Xs.shape[1], "dim_to": Ys.shape[1],
               "cka": _cka_centered(Xs, Ys),
               "r2": fit["r2"],
               "residual_mean": fit["residual_mean"],
               "residual_median": fit["residual_median"],
               "residual_max": fit["residual_max"],
               "rank_from": fit["rank"],
               "raw_distance_mean": float(dist.mean()) if same_width else np.nan,
               "raw_distance_median": float(np.median(dist)) if same_width else np.nan,
               "raw_distance_max": float(dist.max()) if same_width else np.nan}
        rows.append(row)
        if verbose:
            print(f"[{i}/{len(pairs)}] {a} -> {b}: CKA {row['cka']:.3f}  R2 {row['r2']:.3f}  "
                  f"residual mean {row['residual_mean']:.3f}")
    return pd.DataFrame(rows)


def similarity_matrix(table: pd.DataFrame, metric: str = "cka",
                      keys: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Pivot one metric of a ``compare_embeddings`` table into a square DataFrame.

    Rows are ``from``, columns ``to``. The diagonal is filled with the metric's value for
    an embedding against itself (1 for cka and r2, 0 for residuals and distances)."""
    if metric not in METRICS:
        raise KeyError(f"metric must be one of {METRICS}, got {metric!r}")
    if keys is None:
        keys = list(pd.unique(pd.concat([table["from"], table["to"]], ignore_index=True)))
    m = table.pivot(index="from", columns="to", values=metric).reindex(index=keys, columns=keys)
    for k in keys:
        m.loc[k, k] = _SELF_VALUE[metric]
    m.index.name, m.columns.name = "from", "to"
    return m.astype(float)


# ---------------------------------------------------------------------------
# Original formulations, kept for comparison
# ---------------------------------------------------------------------------
# These reproduce the exploration that preceded this module, with names made consistent
# and typos fixed. They differ from the functions above in ways that matter when the
# numbers are compared side by side:
#   * lstsq_measure scales each embedding by its mean per-cell norm but does NOT center it,
#     so a shift between two embeddings stays in its residual. linear_fit centers first.
#   * iterative_mapping minimises the MEAN PER-CELL NORM of the residual,
#     mean_i ||y_i - x_i W||, by plain gradient descent from the identity. linear_fit
#     minimises the SUM OF SQUARES in closed form. The two objectives have different optima
#     (the norm objective weights outlying cells less), and a fixed number of steps at a
#     fixed step size need not reach either.
#   * The transform is (d_from, d_to), initialised to a rectangular identity; the original
#     hard-coded a 10 x 10 identity. For equal widths the start is the same.
#   * residual_stats drops the original "mode" of the residual norms: on continuous values
#     every value is distinct, so the mode is just the minimum.


def scale_by_mean_norm(X, name: str = "embedding") -> np.ndarray:
    """Divide by the mean per-cell L2 norm without centering (the original lstsq preprocessing)."""
    X = _dense_float64(X, name)
    scale = np.linalg.norm(X, axis=1).mean()
    return X / scale if scale > 0 else X


def residual_stats(norms) -> dict:
    """mean, var, std, min, max and median of per-cell residual norms, keyed ``residual_*``."""
    norms = np.asarray(norms, dtype=np.float64)
    return {"residual_mean": float(norms.mean()), "residual_var": float(norms.var()),
            "residual_std": float(norms.std()), "residual_min": float(norms.min()),
            "residual_max": float(norms.max()), "residual_median": float(np.median(norms))}


def measure(X, Y, transform) -> tuple:
    """Residuals of a given linear map X @ transform -> Y.

    Returns
    -------
    residuals      : ndarray (n_cells, d_to), Y − X @ transform
    residual_norms : ndarray (n_cells,), per-cell L2 norm of the residual
    stats          : dict, ``residual_stats`` of those norms"""
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    _check_rows(X, Y)
    residuals = Y - X @ np.asarray(transform, dtype=np.float64)
    norms = np.linalg.norm(residuals, axis=1)
    return residuals, norms, residual_stats(norms)


def iterative_mapping(X, Y, max_iter: int = 500, lr: float = 0.01, device: str = "cpu"):
    """Gradient descent on the mean per-cell residual norm of a linear map X -> Y (needs torch).

    Starts from the identity (rectangular if the widths differ) and takes ``max_iter`` plain
    gradient steps of size ``lr`` on mean_i ||y_i − x_i W||, in float32 as the original did.
    Inputs are used as given; ``iterative_map_test`` standardizes them first.

    Returns
    -------
    loss_curve : ndarray (max_iter,), the loss before each step
    transform  : ndarray (d_from, d_to), the map after the last step"""
    import torch
    X, Y = _dense_float64(X, "X"), _dense_float64(Y, "Y")
    _check_rows(X, Y)
    Xt = torch.as_tensor(X.astype(np.float32), device=device)
    Yt = torch.as_tensor(Y.astype(np.float32), device=device)
    W = torch.eye(Xt.shape[1], Yt.shape[1], device=device, requires_grad=True)
    loss_curve = np.empty(max_iter)
    for i in range(max_iter):
        loss = torch.linalg.norm(Xt @ W - Yt, dim=1).mean()
        loss.backward()
        loss_curve[i] = loss.item()
        with torch.no_grad():
            W -= lr * W.grad
            W.grad = None
    return loss_curve, W.detach().cpu().numpy().astype(np.float64)


def iterative_map_test(embeddings, keys: Optional[Sequence[str]] = None, max_iter: int = 500,
                       lr: float = 0.01, max_cells: Optional[int] = None, random_state: int = 0,
                       device: str = "cpu", return_arrays: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
    """``iterative_mapping`` for every ordered pair of standardized embeddings.

    Same inputs as ``compare_embeddings``. Returns one row per ordered pair with columns
        from, to, n_cells, dim_from, dim_to, n_iter, lr, initial_loss, final_loss,
        residual_mean, residual_var, residual_std, residual_min, residual_max, residual_median
    where initial_loss is the loss at the identity start, final_loss the loss before the
    last step, and the residual_* columns come from ``measure`` on the final map. With
    ``return_arrays=True`` the object columns loss_curve, transform, residual_norms and
    residuals are added (large: residuals is n_cells × d_to per pair)."""
    keys, mats, n_cells = _prepare(embeddings, keys, max_cells, random_state)
    std = {k: standardize(m, k) for k, m in mats.items()}
    del mats
    pairs = _ordered_pairs(keys)
    rows = []
    for i, (a, b) in enumerate(pairs, 1):
        loss_curve, W = iterative_mapping(std[a], std[b], max_iter=max_iter, lr=lr, device=device)
        residuals, norms, stats = measure(std[a], std[b], W)
        row = {"from": a, "to": b, "n_cells": n_cells, "dim_from": std[a].shape[1], "dim_to": std[b].shape[1],
               "n_iter": max_iter, "lr": lr,
               "initial_loss": float(loss_curve[0]), "final_loss": float(loss_curve[-1]), **stats}
        if return_arrays:
            row.update(loss_curve=loss_curve, transform=W, residual_norms=norms, residuals=residuals)
        rows.append(row)
        if verbose:
            print(f"[{i}/{len(pairs)}] iterative {a} -> {b}: loss {loss_curve[0]:.3f} -> {loss_curve[-1]:.3f}  "
                  f"residual mean {stats['residual_mean']:.3f}")
    return pd.DataFrame(rows)


def lstsq_measure(embeddings, keys: Optional[Sequence[str]] = None, max_cells: Optional[int] = None,
                  random_state: int = 0, return_arrays: bool = False,
                  verbose: bool = True) -> pd.DataFrame:
    """Original least-squares map for every ordered pair, on scaled but uncentered embeddings.

    Same inputs as ``compare_embeddings``. Returns one row per ordered pair with columns
        from, to, n_cells, dim_from, dim_to, residual_ss_total, residual_ss_mean, rank,
        singular_max, singular_min, residual_mean, residual_var, residual_std, residual_min,
        residual_max, residual_median
    residual_ss_total is the sum of squared residuals over all cells and target columns and
    residual_ss_mean that sum averaged over target columns (the original "Residuals Mean").
    Both are computed from the residuals directly, so they are defined when the predictor
    is rank-deficient (``np.linalg.lstsq`` returns an empty array there). rank and the
    singular values describe the scaled predictor. With ``return_arrays=True`` the object
    columns transform, singular_values, residual_norms and residuals are added."""
    keys, mats, n_cells = _prepare(embeddings, keys, max_cells, random_state)
    scaled = {k: scale_by_mean_norm(m, k) for k, m in mats.items()}
    del mats
    pairs = _ordered_pairs(keys)
    rows = []
    for i, (a, b) in enumerate(pairs, 1):
        W, _, rank, s = np.linalg.lstsq(scaled[a], scaled[b], rcond=None)
        residuals, norms, stats = measure(scaled[a], scaled[b], W)
        ss = (residuals ** 2).sum(axis=0)
        row = {"from": a, "to": b, "n_cells": n_cells, "dim_from": scaled[a].shape[1], "dim_to": scaled[b].shape[1],
               "residual_ss_total": float(ss.sum()), "residual_ss_mean": float(ss.mean()), "rank": int(rank),
               "singular_max": float(s.max()), "singular_min": float(s.min()), **stats}
        if return_arrays:
            row.update(transform=W, singular_values=s, residual_norms=norms, residuals=residuals)
        rows.append(row)
        if verbose:
            print(f"[{i}/{len(pairs)}] lstsq {a} -> {b}: residual mean {stats['residual_mean']:.3f}  rank {rank}")
    return pd.DataFrame(rows)


def compare_formulations(embeddings, keys: Optional[Sequence[str]] = None,
                         max_cells: Optional[int] = None, random_state: int = 0,
                         max_iter: int = 500, lr: float = 0.01, device: str = "cpu",
                         verbose: bool = True) -> pd.DataFrame:
    """The new metrics and both original formulations on the same cells, in one table.

    Columns are those of ``compare_embeddings`` plus, from ``lstsq_measure``,
    lstsq_residual_mean, lstsq_residual_median and lstsq_rank, and from
    ``iterative_map_test``, iterative_n_iter, iterative_lr, iterative_initial_loss,
    iterative_final_loss, iterative_residual_mean and iterative_residual_median.
    residual_mean (new, centered
    least squares) and iterative_residual_mean (centered gradient descent) are in the same
    units and directly comparable; lstsq_residual_mean is on uncentered inputs."""
    new = compare_embeddings(embeddings, keys, max_cells, random_state, verbose)
    ls = lstsq_measure(embeddings, keys, max_cells, random_state, verbose=verbose)
    it = iterative_map_test(embeddings, keys, max_iter, lr, max_cells, random_state, device, verbose=verbose)
    ls = ls[["from", "to", "residual_mean", "residual_median", "rank"]]
    it = it[["from", "to", "n_iter", "lr", "initial_loss", "final_loss", "residual_mean", "residual_median"]]
    ls = ls.rename(columns={c: f"lstsq_{c}" for c in ls.columns if c not in ("from", "to")})
    it = it.rename(columns={c: f"iterative_{c}" for c in it.columns if c not in ("from", "to")})
    return new.merge(ls, on=["from", "to"]).merge(it, on=["from", "to"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(matrix: pd.DataFrame, ax: Optional[plt.Axes] = None,
                            cmap: str = "viridis", vmin: Optional[float] = None,
                            vmax: Optional[float] = None, annotate: bool = True,
                            fmt: str = "{:.2f}", title: Optional[str] = None,
                            cbar_label: Optional[str] = None):
    """Heatmap of a ``similarity_matrix`` output. Returns the Axes.

    Cells are annotated with their value in black or white, whichever contrasts with the
    colormap at that value. NaN cells (e.g. raw distance across unequal widths) are blank."""
    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(matrix.columns) + 2.0, 0.6 * len(matrix.index) + 1.5))
    vals = matrix.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "")
    if annotate:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = im.cmap(im.norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if lum < 0.5 else "black")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label or "")
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _read_obsm(path: str, keys: Optional[Sequence[str]]) -> dict:
    """Read only the requested obsm entries from an h5ad, leaving X on disk."""
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:                                   # anndata < 0.11
        from anndata.experimental import read_elem
    with h5py.File(path, "r") as f:
        available = list(f["obsm"].keys())
        keys = available if keys is None else list(keys)
        missing = [k for k in keys if k not in available]
        if missing:
            raise KeyError(f"obsm keys not in {path}: {missing}; available: {available}")
        return {k: read_elem(f["obsm"][k]) for k in keys}


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m scboost_ensemble.embedding_metrics",
        description="Pairwise similarity (CKA, linear-map R², residuals) between obsm embeddings.")
    p.add_argument("input", help="h5ad with one embedding per obsm key")
    p.add_argument("--obsm_keys", nargs="+", default=None,
                   help="obsm keys to compare (default: every key; exclude concat keys by listing the rest)")
    p.add_argument("--out", required=True, help="output TSV, one row per ordered pair")
    p.add_argument("--heatmap", default=None, help="optional figure path (png/pdf) for --heatmap_metric")
    p.add_argument("--heatmap_metric", default="cka", choices=METRICS)
    p.add_argument("--max_cells", type=int, default=None, help="subsample this many cells first")
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--originals", action="store_true",
                   help="also run the original lstsq_measure and iterative_map_test on the same cells "
                        "(compare_formulations); adds lstsq_* and iterative_* columns")
    p.add_argument("--max_iter", type=int, default=500, help="iterative_mapping steps (with --originals)")
    p.add_argument("--lr", type=float, default=0.01, help="iterative_mapping step size (with --originals)")
    p.add_argument("--device", default="cpu", help="torch device for iterative_mapping (with --originals)")
    args = p.parse_args(argv)

    embeddings = _read_obsm(args.input, args.obsm_keys)
    print(f"{len(embeddings)} embeddings -> {len(embeddings) * (len(embeddings) - 1)} ordered pairs")
    if args.originals:
        table = compare_formulations(embeddings, max_cells=args.max_cells, random_state=args.random_state,
                                     max_iter=args.max_iter, lr=args.lr, device=args.device)
    else:
        table = compare_embeddings(embeddings, max_cells=args.max_cells, random_state=args.random_state)
    table.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(table)} pairs to {args.out}")
    print("\nLinear CKA (symmetric):")
    print(similarity_matrix(table, "cka").round(3).to_string())
    if args.heatmap:
        ax = plot_similarity_heatmap(similarity_matrix(table, args.heatmap_metric),
                                     title=args.heatmap_metric, cbar_label=args.heatmap_metric)
        ax.figure.savefig(args.heatmap, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Wrote {args.heatmap}")


if __name__ == "__main__":
    main()
