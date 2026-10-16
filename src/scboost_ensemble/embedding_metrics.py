"""Pairwise similarity between the embeddings in ``adata.obsm``.

scBoost stacks classifiers trained on several embeddings of the same cells, which only helps
where the embeddings differ. These functions measure, without labels, how redundant two of
them are. Each embedding is first scaled to mean per-cell norm 1, so every residual is in
units of "a typical cell's distance from the centroid" and comparable across datasets.

New metrics
    linear_cka(X, Y)       linear centered kernel alignment (Kornblith et al. 2019): symmetric,
                           in [0, 1], invariant to rotation, scale and shift; any widths.
    linear_fit(X, Y)       closed-form least-squares map X -> Y on centered inputs: R² and
                           per-cell residuals. Directional.
    paired_distance(X, Y)  per-cell distance between the standardized coordinates; equal widths
                           only, and meaningful only when the axes already correspond.

Original formulations, kept for comparison
    lstsq_measure          least squares on scaled but *uncentered* inputs, so a mean offset
                           between two embeddings is counted as residual.
    iterative_mapping      gradient descent on mean_i ||y_i - x_i W|| from the identity (torch);
                           a fixed number of steps need not reach the optimum.

All-pairs tables: compare_embeddings (new), lstsq_measure, iterative_map_test (originals) and
compare_formulations (all three merged). similarity_matrix pivots one column to a square table,
cluster_order orders it by hierarchical clustering, plot_similarity_heatmap draws it.

CLI: python -m scboost_ensemble.embedding_metrics in.h5ad --obsm_keys A B C --out pairs.tsv
     [--heatmap cka.png] [--max_cells N] [--originals --max_iter 500 --lr 0.01 --device cuda]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

METRICS = ("cka", "r2", "residual_mean", "residual_median", "residual_max",
           "raw_distance_mean", "raw_distance_median", "raw_distance_max")


# -- preprocessing ------------------------------------------------------------

def _as_array(X, name="embedding"):
    """Dense float64 (n_cells, d) copy, or ValueError."""
    X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name}: expected (n_cells, d), got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: contains NaN or inf")
    return X


def _same_cells(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"embeddings must describe the same cells: {X.shape[0]} vs {Y.shape[0]} rows")


def center(X, name="embedding"):
    """Columns centered at 0."""
    X = _as_array(X, name)
    return X - X.mean(axis=0)


def scale_by_mean_norm(X, name="embedding"):
    """Divided by the mean per-cell norm; not centered (the original lstsq preprocessing)."""
    X = _as_array(X, name)
    s = np.linalg.norm(X, axis=1).mean()
    return X / s if s > 0 else X


def standardize(X, name="embedding"):
    """Centered, then scaled to mean per-cell norm 1."""
    return scale_by_mean_norm(center(X, name), name)


# -- pairwise metrics ---------------------------------------------------------

def linear_cka(X, Y):
    """||Xc'Yc||²_F / (||Xc'Xc||_F ||Yc'Yc||_F) on column-centered X, Y; NaN if either is constant."""
    Xc, Yc = center(X, "X"), center(Y, "Y")
    _same_cells(Xc, Yc)
    nx, ny = np.linalg.norm(Xc.T @ Xc), np.linalg.norm(Yc.T @ Yc)
    return float(np.linalg.norm(Xc.T @ Yc) ** 2 / (nx * ny)) if nx and ny else float("nan")


def residual_stats(norms):
    """mean, var, std, min, max, median of per-cell residual norms, keyed residual_*."""
    n = np.asarray(norms, dtype=np.float64)
    return {"residual_mean": n.mean(), "residual_var": n.var(), "residual_std": n.std(),
            "residual_min": n.min(), "residual_max": n.max(), "residual_median": np.median(n)}


def measure(X, Y, transform):
    """Residuals of a given map: (Y - X @ transform, their per-cell norms, residual_stats of those)."""
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    _same_cells(X, Y)
    residuals = Y - X @ np.asarray(transform, dtype=np.float64)
    norms = np.linalg.norm(residuals, axis=1)
    return residuals, norms, residual_stats(norms)


def _lstsq(X, Y):
    """Least-squares map X -> Y: r2, residual_*, rank of X, coef (d_x, d_y), residual_norms."""
    coef, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    residuals, norms, stats = measure(X, Y, coef)
    ss_tot = (Y ** 2).sum()
    r2 = 1 - (residuals ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": float(r2), **stats, "rank": int(rank), "coef": coef, "residual_norms": norms}


def linear_fit(X, Y):
    """Closed-form least-squares map X -> Y after standardizing both (no intercept needed).
    Returns r2 (fraction of Y's variance explained), residual_* summaries, rank, coef, residual_norms."""
    Xs, Ys = standardize(X, "X"), standardize(Y, "Y")
    _same_cells(Xs, Ys)
    return _lstsq(Xs, Ys)


def paired_distance(X, Y):
    """Per-cell L2 distance between the standardized coordinates; equal widths only."""
    Xs, Ys = standardize(X, "X"), standardize(Y, "Y")
    _same_cells(Xs, Ys)
    if Xs.shape[1] != Ys.shape[1]:
        raise ValueError(f"paired_distance needs equal width, got {Xs.shape[1]} and {Ys.shape[1]}")
    return np.linalg.norm(Xs - Ys, axis=1)


def iterative_mapping(X, Y, max_iter=500, lr=0.01, device="cpu"):
    """Original gradient descent on mean_i ||y_i - x_i W|| from the (rectangular) identity, float32.
    Inputs are used as given. Returns (loss before each step, final W of shape (d_x, d_y))."""
    import torch
    X, Y = _as_array(X, "X"), _as_array(Y, "Y")
    _same_cells(X, Y)
    Xt, Yt = (torch.as_tensor(A.astype(np.float32), device=device) for A in (X, Y))
    W = torch.eye(X.shape[1], Y.shape[1], device=device, requires_grad=True)
    losses = np.empty(max_iter)
    for i in range(max_iter):
        loss = torch.linalg.norm(Xt @ W - Yt, dim=1).mean()
        loss.backward()
        losses[i] = loss.item()
        with torch.no_grad():
            W -= lr * W.grad
            W.grad = None
    return losses, W.detach().cpu().numpy().astype(np.float64)


# -- all ordered pairs --------------------------------------------------------

def _load(embeddings, keys, max_cells, random_state):
    """{key: float64 array} for the requested keys (default: all), same subsample of cells for every key."""
    source = embeddings.obsm if hasattr(embeddings, "obsm") else embeddings
    keys = list(source.keys()) if keys is None else list(keys)
    missing = [k for k in keys if k not in source]
    if missing:
        raise KeyError(f"embedding keys not found: {missing}; available: {list(source.keys())}")
    if len(keys) < 2:
        raise ValueError("need at least two embeddings to compare")
    mats = {k: _as_array(source[k], k) for k in keys}
    n = {m.shape[0] for m in mats.values()}
    if len(n) != 1:
        raise ValueError(f"all embeddings must have the same number of cells, got {sorted(n)}")
    n_cells = n.pop()
    if max_cells is not None and n_cells > max_cells:
        idx = np.sort(np.random.default_rng(random_state).choice(n_cells, max_cells, replace=False))
        mats = {k: m[idx] for k, m in mats.items()}
    return mats


def _per_pair(embeddings, keys, max_cells, random_state, prep, metrics, label, verbose):
    """One row per ordered pair (a, b): from, to, n_cells, dim_from, dim_to, then metrics(prep(a), prep(b))."""
    mats = {k: prep(m, k) for k, m in _load(embeddings, keys, max_cells, random_state).items()}
    pairs = [(a, b) for a in mats for b in mats if a != b]
    rows = []
    for i, (a, b) in enumerate(pairs, 1):
        X, Y = mats[a], mats[b]
        rows.append({"from": a, "to": b, "n_cells": X.shape[0], "dim_from": X.shape[1], "dim_to": Y.shape[1],
                     **metrics(X, Y)})
        if verbose:
            shown = {k: rows[-1][k] for k in ("cka", "r2", "final_loss", "residual_mean") if k in rows[-1]}
            print(f"[{i}/{len(pairs)}] {label} {a} -> {b}: " + "  ".join(f"{k} {v:.3f}" for k, v in shown.items()))
    return pd.DataFrame(rows)


def compare_embeddings(embeddings, keys=None, max_cells=None, random_state=0, verbose=True):
    """New metrics for every ordered pair: cka, r2, residual_{mean,median,max}, rank_from and, for equal
    widths, raw_distance_{mean,median,max} (NaN otherwise). cka is symmetric; r2 and residuals are from -> to."""
    def metrics(X, Y):
        fit = _lstsq(X, Y)
        d = np.linalg.norm(X - Y, axis=1) if X.shape[1] == Y.shape[1] else np.array([np.nan])
        return {"cka": linear_cka(X, Y), "r2": fit["r2"], "residual_mean": fit["residual_mean"],
                "residual_median": fit["residual_median"], "residual_max": fit["residual_max"],
                "rank_from": fit["rank"], "raw_distance_mean": d.mean(), "raw_distance_median": np.median(d),
                "raw_distance_max": d.max()}
    return _per_pair(embeddings, keys, max_cells, random_state, standardize, metrics, "new", verbose)


def lstsq_measure(embeddings, keys=None, max_cells=None, random_state=0, return_arrays=False, verbose=True):
    """Original least squares for every ordered pair on scaled, uncentered inputs: residual_ss_total and
    residual_ss_mean (squared residuals summed over all cells, in total and per target column), rank and
    singular_{max,min} of the predictor, residual_* summaries; with return_arrays also transform,
    singular_values, residual_norms, residuals."""
    def metrics(X, Y):
        W, _, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        residuals, norms, stats = measure(X, Y, W)
        ss = (residuals ** 2).sum(axis=0)
        row = {"residual_ss_total": ss.sum(), "residual_ss_mean": ss.mean(), "rank": int(rank),
               "singular_max": s.max(), "singular_min": s.min(), **stats}
        if return_arrays:
            row.update(transform=W, singular_values=s, residual_norms=norms, residuals=residuals)
        return row
    return _per_pair(embeddings, keys, max_cells, random_state, scale_by_mean_norm, metrics, "lstsq", verbose)


def iterative_map_test(embeddings, keys=None, max_iter=500, lr=0.01, max_cells=None, random_state=0,
                       device="cpu", return_arrays=False, verbose=True):
    """Original gradient descent for every ordered pair on standardized inputs: n_iter, lr, initial_loss
    (at the identity), final_loss (before the last step), residual_* of the final map; with return_arrays
    also loss_curve, transform, residual_norms, residuals."""
    def metrics(X, Y):
        losses, W = iterative_mapping(X, Y, max_iter, lr, device)
        residuals, norms, stats = measure(X, Y, W)
        row = {"n_iter": max_iter, "lr": lr, "initial_loss": losses[0], "final_loss": losses[-1], **stats}
        if return_arrays:
            row.update(loss_curve=losses, transform=W, residual_norms=norms, residuals=residuals)
        return row
    return _per_pair(embeddings, keys, max_cells, random_state, standardize, metrics, "iterative", verbose)


def compare_formulations(embeddings, keys=None, max_cells=None, random_state=0, max_iter=500, lr=0.01,
                         device="cpu", verbose=True):
    """compare_embeddings plus both originals on the same cells, merged on (from, to): lstsq_{residual_mean,
    residual_median,rank} and iterative_{n_iter,lr,initial_loss,final_loss,residual_mean,residual_median}."""
    table = compare_embeddings(embeddings, keys, max_cells, random_state, verbose)
    extra = {"lstsq": (lstsq_measure(embeddings, keys, max_cells, random_state, verbose=verbose),
                       ["residual_mean", "residual_median", "rank"]),
             "iterative": (iterative_map_test(embeddings, keys, max_iter, lr, max_cells, random_state, device,
                                              verbose=verbose),
                           ["n_iter", "lr", "initial_loss", "final_loss", "residual_mean", "residual_median"])}
    for name, (t, cols) in extra.items():
        table = table.merge(t[["from", "to"] + cols].rename(columns={c: f"{name}_{c}" for c in cols}),
                            on=["from", "to"])
    return table


# -- square tables and plots --------------------------------------------------

def similarity_matrix(table, metric="cka", keys=None):
    """Pivot one column of an all-pairs table to a square (from x to) DataFrame.
    The diagonal is the self value: 1 for cka and r2, 0 for residuals and distances."""
    if metric not in table.columns:
        raise KeyError(f"{metric!r} is not a column of the table")
    keys = list(keys) if keys is not None else list(pd.unique(pd.concat([table["from"], table["to"]])))
    m = table.pivot(index="from", columns="to", values=metric).reindex(index=keys, columns=keys).astype(float)
    for k in keys:
        m.loc[k, k] = 1.0 if metric.endswith(("cka", "r2")) else 0.0
    m.index.name, m.columns.name = "from", "to"
    return m


def cluster_order(matrix, method="average"):
    """Labels of a square table in hierarchical-clustering leaf order. The table is symmetrized and read
    as distances; a similarity table (diagonal 1) is first turned into 1 - value."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    d = matrix.to_numpy(dtype=float)
    d = (d + d.T) / 2
    if np.allclose(np.diag(d), 1):
        d = 1 - d
    np.fill_diagonal(d, 0)
    order = leaves_list(linkage(squareform(d, checks=False), method=method))
    return [matrix.index[i] for i in order]


def plot_similarity_heatmap(matrix, ax=None, cmap="viridis", vmin=None, vmax=None, annotate=True,
                            fmt="{:.2f}", title=None, cbar_label=None, cluster=False):
    """Annotated heatmap of a square table; returns the Axes. cluster=True reorders rows and columns
    by cluster_order. Labels are white or black, whichever contrasts with the cell colour; NaN cells are blank."""
    if cluster:
        order = cluster_order(matrix)
        matrix = matrix.loc[order, order]
    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(matrix.columns) + 2.0, 0.6 * len(matrix.index) + 1.5))
    vals = matrix.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "")
    if annotate:
        for (i, j), v in np.ndenumerate(vals):
            if np.isfinite(v):
                r, g, b, _ = im.cmap(im.norm(v))
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label or "")
    if title:
        ax.set_title(title)
    return ax


# -- command line -------------------------------------------------------------

def _read_obsm(path, keys):
    """Only the requested obsm entries of an h5ad (default: all); X stays on disk."""
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
    p = argparse.ArgumentParser(prog="python -m scboost_ensemble.embedding_metrics",
                                description="Pairwise similarity (CKA, linear-map R², residuals) between obsm embeddings.")
    p.add_argument("input", help="h5ad with one embedding per obsm key")
    p.add_argument("--obsm_keys", nargs="+", default=None, help="keys to compare (default: every obsm key)")
    p.add_argument("--out", required=True, help="output TSV, one row per ordered pair")
    p.add_argument("--heatmap", default=None, help="optional figure (png/pdf) of --heatmap_metric")
    p.add_argument("--heatmap_metric", default="cka", choices=METRICS)
    p.add_argument("--max_cells", type=int, default=None, help="subsample this many cells first")
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--originals", action="store_true",
                   help="also run the original lstsq_measure and iterative_map_test (adds lstsq_* and iterative_* columns)")
    p.add_argument("--max_iter", type=int, default=500, help="gradient-descent steps (with --originals)")
    p.add_argument("--lr", type=float, default=0.01, help="gradient-descent step size (with --originals)")
    p.add_argument("--device", default="cpu", help="torch device for the gradient descent (with --originals)")
    a = p.parse_args(argv)

    emb = _read_obsm(a.input, a.obsm_keys)
    print(f"{len(emb)} embeddings -> {len(emb) * (len(emb) - 1)} ordered pairs")
    if a.originals:
        table = compare_formulations(emb, max_cells=a.max_cells, random_state=a.random_state,
                                     max_iter=a.max_iter, lr=a.lr, device=a.device)
    else:
        table = compare_embeddings(emb, max_cells=a.max_cells, random_state=a.random_state)
    table.to_csv(a.out, sep="\t", index=False)
    print(f"Wrote {len(table)} pairs to {a.out}\n\nLinear CKA (symmetric):")
    print(similarity_matrix(table, "cka").round(3).to_string())
    if a.heatmap:
        ax = plot_similarity_heatmap(similarity_matrix(table, a.heatmap_metric), title=a.heatmap_metric,
                                     cbar_label=a.heatmap_metric)
        ax.figure.savefig(a.heatmap, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Wrote {a.heatmap}")


if __name__ == "__main__":
    main()
