"""Plotting utilities for scboost-ensemble.

Each function returns a matplotlib Axes (or list of Axes) so plots can be
composed into multi-panel figures or styled by the caller.

Functions
---------
plot_method_importance         — bar chart of meta-learner importance per embedding
plot_training_summary          — best iteration per fold for base + meta learners
plot_classification_report     — per-cell-type accuracy bars (from a saved report TSV)
plot_method_comparison         — generic multi-arm bar chart (compare e.g. popV vs scBoost)
plot_oof_meta_features         — UMAP/PCA scatter of OOF meta-features colored by cell type
"""

from __future__ import annotations
from typing import Optional, Iterable, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Method importance
# ---------------------------------------------------------------------------

def plot_method_importance(stacker, ax: Optional[plt.Axes] = None,
                            color: str = "#1C9099", title: Optional[str] = None):
    """Horizontal bar chart of per-embedding importance from the meta-learners."""
    imp = stacker.get_method_importance()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2, 0.4 * len(imp))))
    methods, values = zip(*imp.items())
    ypos = list(range(len(methods)))[::-1]
    ax.barh(ypos, values, color=color, edgecolor="black", linewidth=0.4)
    for y, v in zip(ypos, values):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Method importance (mean across meta-learners)")
    ax.set_title(title or "Method importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ---------------------------------------------------------------------------
# 2. Training summary (best iteration per fold)
# ---------------------------------------------------------------------------

def plot_training_summary(stacker, ax: Optional[plt.Axes] = None):
    """Line plot: best iteration per fold for each base learner + meta-learners.
    Useful for sanity-checking early stopping behavior."""
    summary = stacker.get_training_summary()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    n_folds = summary["n_folds"]
    x = range(n_folds)
    for key, iters in summary["base_best_iterations"].items():
        ax.plot(x, iters, marker="o", label=f"base: {key}", alpha=0.7)
    ax.plot(x, summary["meta_best_iterations"], marker="s", color="black",
            linewidth=2, label="meta", zorder=10)
    ax.set_xlabel("Fold index")
    ax.set_ylabel("Best iteration (early stopping)")
    ax.set_title("Training summary — best iteration per fold")
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ---------------------------------------------------------------------------
# 3. Classification report (per cell-type accuracy bars)
# ---------------------------------------------------------------------------

def plot_classification_report(report_df: Union[pd.DataFrame, str],
                                top_n: Optional[int] = 20,
                                sort: str = "f1-score",
                                ax: Optional[plt.Axes] = None):
    """Bar chart of per-cell-type F1 (or precision/recall) from a classification
    report. Accepts a DataFrame or a TSV path."""
    if isinstance(report_df, (str, Path)):
        report_df = pd.read_csv(report_df, sep="\t", index_col=0)
    # Drop aggregate rows
    df = report_df.drop(index=[r for r in ("accuracy", "macro avg", "weighted avg",
                                            "micro avg")
                                if r in report_df.index], errors="ignore")
    df = df.sort_values(sort, ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(df))))
    ypos = list(range(len(df)))[::-1]
    ax.barh(ypos, df[sort], color="#5DA5DA", edgecolor="black", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel(sort)
    ax.set_title(f"Per-cell-type {sort} (top {top_n})" if top_n else f"Per-cell-type {sort}")
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ---------------------------------------------------------------------------
# 4. Generic multi-arm comparison
# ---------------------------------------------------------------------------

def plot_method_comparison(
    metrics: Union[pd.DataFrame, str],
    arm_col: str = "arm",
    split_col: Optional[str] = "split",
    value_col: str = "top1_accuracy",
    orientation: str = "horizontal",
    ax: Optional[plt.Axes] = None,
):
    """Bar chart comparing multiple arms (one bar per arm) on top-1 accuracy.

    If `split_col` is non-None, produces one panel per split (a list of Axes
    is returned). Otherwise returns a single Axes.
    """
    if isinstance(metrics, (str, Path)):
        metrics = pd.read_csv(metrics, sep="\t")
    if split_col is None or split_col not in metrics.columns:
        return _plot_arm_bars(metrics, arm_col, value_col, orientation, ax)
    splits = list(metrics[split_col].unique())
    if ax is None:
        fig, axes = plt.subplots(1, len(splits),
                                  figsize=(6 * len(splits), max(4, 0.4 * len(metrics[arm_col].unique()))))
        if len(splits) == 1:
            axes = [axes]
    else:
        axes = [ax]
    for a, split in zip(axes, splits):
        sub = metrics[metrics[split_col] == split]
        _plot_arm_bars(sub, arm_col, value_col, orientation, a)
        a.set_title(split)
    return axes


def _plot_arm_bars(df, arm_col, value_col, orientation, ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(df))))
    if orientation == "horizontal":
        ypos = list(range(len(df)))[::-1]
        ax.barh(ypos, df[value_col], color="#1C9099", edgecolor="black", linewidth=0.4)
        for y, v in zip(ypos, df[value_col]):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(df[arm_col], fontsize=7)
        ax.set_xlabel(value_col)
        ax.set_xlim(0, 1.0)
    else:
        xpos = list(range(len(df)))
        ax.bar(xpos, df[value_col], color="#1C9099", edgecolor="black", linewidth=0.4)
        for x, v in zip(xpos, df[value_col]):
            ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(xpos)
        ax.set_xticklabels(df[arm_col], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(value_col)
        ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ---------------------------------------------------------------------------
# 5. OOF meta-features diagnostic
# ---------------------------------------------------------------------------

def plot_oof_meta_features(
    stacker, adata,
    max_points: int = 2000,
    projection: str = "auto",
    ax: Optional[plt.Axes] = None,
    random_state: int = 42,
):
    """Project the (n_train, n_methods × n_classes) OOF meta-feature matrix to 2D
    and scatter colored by true cell type.

    Helps diagnose:
      - whether base learners already separate cell types (clean clusters → easy meta job)
      - hard cells (off-cluster outliers)
      - cells that all base learners disagreed on

    Parameters
    ----------
    stacker : fitted StackingEnsembleDonorMetaCached
    adata : AnnData
        Same adata that was passed to ``fit()``. We rebuild the OOF matrix
        from the cached base classifiers' per-cell predictions.
    max_points : int
        Subsample for speed; set to None for all cells.
    projection : {"auto", "umap", "pca"}
        "auto" picks umap if umap-learn is installed, else pca.
    """
    if stacker.base_classifiers_ is None or not stacker.base_classifiers_:
        raise RuntimeError("stacker must be fit() first")

    # Recompute OOF matrix by aggregating per-fold predictions on adata
    embs, labels, batch, _ = stacker._extract(adata)
    enc_labels = stacker.label_encoder_.transform(labels)
    fold_assignments = stacker.fold_assignments_
    n = len(enc_labels)
    C = stacker.n_cell_types
    n_methods = len(stacker.obsm_keys)
    meta = np.zeros((n, n_methods * C), dtype=np.float32)
    for k in range(stacker.n_folds):
        in_fold_k = (fold_assignments == k)
        if not in_fold_k.any():
            continue
        for ki, key in enumerate(stacker.obsm_keys):
            clf = stacker.base_classifiers_[key][k]
            cs, ce = ki * C, (ki + 1) * C
            meta[in_fold_k, cs:ce] = clf.predict_proba(embs[key][in_fold_k]).astype(np.float32)

    # Optional subsample
    if max_points is not None and n > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        meta = meta[idx]
        labels_sub = labels[idx]
    else:
        labels_sub = labels

    # Project to 2D
    proj = projection
    if proj == "auto":
        try:
            import umap  # noqa: F401
            proj = "umap"
        except ImportError:
            proj = "pca"
    if proj == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            xy = reducer.fit_transform(meta)
        except ImportError as e:
            raise ImportError("umap-learn not installed. Use projection='pca' or "
                              "install with `pip install scboost-ensemble[umap]`") from e
    else:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=random_state).fit_transform(meta)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    unique_labels = pd.unique(labels_sub)
    cmap = plt.cm.tab20
    label_to_color = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}
    colors = [label_to_color[l] for l in labels_sub]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=4, alpha=0.6, linewidths=0)
    ax.set_xlabel(f"{proj.upper()} 1")
    ax.set_ylabel(f"{proj.upper()} 2")
    ax.set_title(f"OOF meta-features ({proj}, {len(labels_sub)} cells)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
