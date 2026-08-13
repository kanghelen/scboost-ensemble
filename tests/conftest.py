"""Shared pytest fixtures.

Tiny synthetic AnnData: 5 donors × 4 cell types × 20 cells per (donor, type) =
400 cells, with two 8-d embeddings in .obsm. Big enough to exercise 5-fold
StratifiedGroupKFold by donor (5 donors → 5 folds, one donor per fold).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import anndata as ad
import pytest


@pytest.fixture(scope="session")
def tiny_adata():
    rng = np.random.default_rng(42)
    n_donors, n_types, n_per = 5, 4, 20
    n = n_donors * n_types * n_per

    donor_ids   = np.repeat(np.arange(n_donors), n_types * n_per)
    cell_types  = np.tile(np.repeat(np.arange(n_types), n_per), n_donors)
    cell_type_names = np.array(["A", "B", "C", "D"])[cell_types]
    donor_names = np.array([f"donor{i}" for i in donor_ids])

    # Two synthetic embeddings: each cell type centered at a distinct location
    centers_A = rng.normal(0, 1, size=(n_types, 8))
    centers_B = rng.normal(0, 1, size=(n_types, 8))
    emb_A = centers_A[cell_types] + rng.normal(0, 0.3, size=(n, 8))
    emb_B = centers_B[cell_types] + rng.normal(0, 0.3, size=(n, 8))

    obs = pd.DataFrame({
        "cell_type": cell_type_names,
        "batch":     donor_names,
        "donor_id":  donor_names,
    }, index=[f"cell_{i:04d}" for i in range(n)])

    adata = ad.AnnData(
        X=rng.random((n, 10), dtype=np.float32),
        obs=obs,
        obsm={"embA": emb_A.astype(np.float32),
              "embB": emb_B.astype(np.float32)},
    )
    adata.obs["split"] = np.where(adata.obs["donor_id"] == "donor4", "test", "train")
    return adata
