"""Hybrid donor folds (scboost_ensemble.folds) and class-balanced losses (scboost_ensemble.balanced)."""

from __future__ import annotations
import numpy as np
import pytest

from scboost_ensemble.folds import (DONOR_DISJOINT, WITHIN_DONOR, check_fold_properties,
                                    hybrid_donor_folds)
from scboost_ensemble.balanced import class_weights


def _labels_donors():
    """3 classes in all 6 donors, plus class R in one donor only."""
    labels = np.repeat(["A", "B", "C"], 120).tolist() + ["R"] * 12
    donors = np.tile(np.repeat([f"d{i}" for i in range(6)], 20), 3).tolist() + ["d0"] * 12
    return np.array(labels), np.array(donors)


def test_hybrid_folds_rare_class_within_donor():
    labels, donors = _labels_donors()
    fold, table = hybrid_donor_folds(labels, donors, n_folds=5)
    assert check_fold_properties(labels, donors, fold, table, n_folds=5)
    assert table.loc["R", "validation"] == WITHIN_DONOR
    assert table.loc["R", "folds"] == 5                       # the rare class reaches every branch
    assert (table.loc[["A", "B", "C"], "validation"] == DONOR_DISJOINT).all()
    common = ~np.isin(labels, ["R"])
    for d in np.unique(donors[common]):                       # a donor's common-class cells sit in one fold
        assert len(np.unique(fold[common & (donors == d)])) == 1


def test_hybrid_folds_are_reproducible():
    labels, donors = _labels_donors()
    f1, _ = hybrid_donor_folds(labels, donors, seed=42)
    f2, _ = hybrid_donor_folds(labels, donors, seed=42)
    assert np.array_equal(f1, f2)


def test_class_weights():
    y = np.array([0] * 90 + [1] * 10)
    assert np.allclose(class_weights(y, 0.0, 2), 1.0)
    w = class_weights(y, 1.0, 2)
    assert np.isclose(w[y].mean(), 1.0)                       # mean weight per cell is 1
    assert np.isclose(90 * w[0], 10 * w[1])                   # beta = 1: equal total weight per class
    w3 = class_weights(np.array([0, 0, 2]), 1.0, 3)
    assert w3[1] == 0                                         # absent class


def test_balanced_beta0_is_the_engine(tiny_adata):
    """beta = 0 takes the engine's own code path, so its predictions equal the unweighted stacker's."""
    pytest.importorskip("torch")
    from scboost_ensemble.reference_model import RefStacker, RefStackerBalanced
    train = tiny_adata[tiny_adata.obs["split"] == "train"].copy()
    test = tiny_adata[tiny_adata.obs["split"] == "test"].copy()
    kw = dict(obsm_keys=["embA", "embB"], label_key="cell_type", batch_key="donor_id", use_batch_feature=False,
              n_folds=4, cv_scheme="donor", device="cpu", base_algo="mlp",
              algo_params=dict(arch="mlp", hidden=[16], dropout=0.1, weight_decay=1e-4, lr=1e-2,
                               batch_size=4096, max_epochs=30))
    p0 = RefStacker(**kw).fit(train).predict_proba(test)
    p1 = RefStackerBalanced(class_weight_beta=0.0, **kw).fit(train).predict_proba(test)
    assert np.array_equal(p0, p1)
    p2 = RefStackerBalanced(class_weight_beta=1.0, **kw).fit(train).predict_proba(test)
    assert p2.shape == p0.shape and np.allclose(p2.sum(1), 1, atol=1e-4)
