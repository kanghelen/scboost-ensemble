"""
Option A: Random stratified train/test split.
Splits cells randomly while preserving cell type and batch proportions.
Best for evaluating classification accuracy on held-out cells from the
same batches present in training.

Writes adata.obs["split"] with values "train" / "test".
"""

import numpy as np
from anndata import AnnData
from sklearn.model_selection import train_test_split
from typing import Optional


def stratified_split(
    adata: AnnData,
    label_key: str = "cell_type",
    batch_key: str = "batch",
    test_size: float = 0.2,
    random_state: int = 42,
    key_added: str = "split",
    stratify_by_batch: bool = True,
):
    """
    Random stratified train/test split on an AnnData object.

    Parameters
    ----------
    adata : AnnData
    label_key : str
        Column in adata.obs with cell type labels.
    batch_key : str
        Column in adata.obs with batch IDs.
    test_size : float
        Fraction of cells for test set.
    random_state : int
        Random seed for reproducibility.
    key_added : str
        Column name written to adata.obs.
    stratify_by_batch : bool
        If True, stratify by both cell type and batch (Option C behavior).
        If False, stratify by cell type only (pure Option A).

    Returns
    -------
    adata : AnnData (modified in place, also returned for convenience)
    """
    if stratify_by_batch and batch_key in adata.obs.columns:
        stratify = (
            adata.obs[label_key].astype(str) + "_" + adata.obs[batch_key].astype(str)
        ).values
        # Fall back to label-only if any group has < 2 members
        _, counts = np.unique(stratify, return_counts=True)
        if counts.min() < 2:
            print("  Some (cell_type, batch) groups have <2 cells; "
                  "falling back to cell-type-only stratification.")
            stratify = adata.obs[label_key].values
    else:
        stratify = adata.obs[label_key].values

    idx_train, idx_test = train_test_split(
        np.arange(adata.n_obs),
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    adata.obs[key_added] = "train"
    adata.obs.iloc[idx_test, adata.obs.columns.get_loc(key_added)] = "test"

    n_train, n_test = len(idx_train), len(idx_test)
    print(f"Split: {n_train} train ({n_train/adata.n_obs:.0%}), "
          f"{n_test} test ({n_test/adata.n_obs:.0%})")
    print(f"  Train cell types: {adata.obs.loc[adata.obs[key_added]=='train', label_key].nunique()}")
    print(f"  Test  cell types: {adata.obs.loc[adata.obs[key_added]=='test', label_key].nunique()}")

    return adata
