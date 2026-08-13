"""Smoke test: fit + predict on a tiny synthetic dataset (CPU, no GPU).
Verifies the noLeak architecture wires together end-to-end."""

from __future__ import annotations
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from scboost_ensemble import StackingEnsembleDonorMetaCached


def test_smoke_fit_predict(tiny_adata):
    train = tiny_adata[tiny_adata.obs["split"] == "train"].copy()
    test  = tiny_adata[tiny_adata.obs["split"] == "test"].copy()

    stacker = StackingEnsembleDonorMetaCached(
        obsm_keys=["embA", "embB"],
        cv_scheme="donor",       # 4 train donors → 4-fold by donor
        n_folds=4,
        early_stopping_rounds=5,
        device="cpu", n_gpus=1,
        use_batch_feature=False,
    )
    stacker.fit(train)
    preds = stacker.predict(test, key_added="pred")
    proba = stacker.predict_proba(test)

    # Sanity checks
    assert proba.shape == (test.n_obs, len(np.unique(train.obs["cell_type"])))
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-3)
    assert len(stacker.meta_classifiers_) == 4

    # On synthetic data with clear class structure, accuracy should be reasonable
    acc = accuracy_score(test.obs["cell_type"], preds)
    assert acc > 0.5, f"smoke test accuracy too low: {acc:.3f}"


def test_method_importance(tiny_adata):
    train = tiny_adata[tiny_adata.obs["split"] == "train"].copy()
    stacker = StackingEnsembleDonorMetaCached(
        obsm_keys=["embA", "embB"], cv_scheme="donor", n_folds=4,
        early_stopping_rounds=5, device="cpu", use_batch_feature=False,
    )
    stacker.fit(train)
    imp = stacker.get_method_importance()
    assert set(imp.keys()) == {"embA", "embB"}
    # Importances sum to ~1
    assert 0.99 < sum(imp.values()) < 1.01


def test_training_summary(tiny_adata):
    train = tiny_adata[tiny_adata.obs["split"] == "train"].copy()
    stacker = StackingEnsembleDonorMetaCached(
        obsm_keys=["embA", "embB"], cv_scheme="donor", n_folds=4,
        early_stopping_rounds=5, device="cpu", use_batch_feature=False,
    )
    stacker.fit(train)
    summary = stacker.get_training_summary()
    assert "base_best_iterations" in summary
    assert "meta_best_iterations" in summary
    assert len(summary["meta_best_iterations"]) == 4


def test_cache_roundtrip(tiny_adata, tmp_path):
    """Cache hit on second fit() call with the same data + dir."""
    train = tiny_adata[tiny_adata.obs["split"] == "train"].copy()
    cache = tmp_path / "cache"

    # First fit — all misses
    s1 = StackingEnsembleDonorMetaCached(
        obsm_keys=["embA", "embB"], cv_scheme="donor", n_folds=4,
        early_stopping_rounds=5, device="cpu", use_batch_feature=False,
        base_cache_dir=str(cache),
    )
    s1.fit(train)
    assert s1.cache_stats_["misses"] == 8   # 2 embeddings × 4 folds
    assert s1.cache_stats_["hits"]   == 0

    # Second fit — all hits
    s2 = StackingEnsembleDonorMetaCached(
        obsm_keys=["embA", "embB"], cv_scheme="donor", n_folds=4,
        early_stopping_rounds=5, device="cpu", use_batch_feature=False,
        base_cache_dir=str(cache),
    )
    s2.fit(train)
    assert s2.cache_stats_["hits"]   == 8
    assert s2.cache_stats_["misses"] == 0
