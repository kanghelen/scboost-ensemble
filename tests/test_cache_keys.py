"""Base-learner cache keying: width isolation, legacy adoption, content checks.

Regression cover for the 0.3.0 fix. Before it, an obsm key that kept its name
while changing width returned a model of the wrong shape from the cache — fit()
succeeded silently on stale OOF features and predict() raised
``ValueError: Feature shape mismatch``.
"""

from __future__ import annotations
import json
import re

import numpy as np

from scboost_ensemble import StackingEnsembleDonorMetaCached
from scboost_ensemble._xgb_utils import GlobalProbsWrapper
from scboost_ensemble.stacking import _model_n_features


def _fit(adata, cache, **kw):
    s = StackingEnsembleDonorMetaCached(
        obsm_keys=["embA", "embB"], cv_scheme="donor", n_folds=4,
        early_stopping_rounds=5, device="cpu", use_batch_feature=False,
        base_cache_dir=str(cache), **kw,
    )
    s.fit(adata[adata.obs["split"] == "train"].copy())
    return s


def _predict(stacker, adata):
    return stacker.predict(adata[adata.obs["split"] == "test"].copy())


def _narrowed(adata, d=4):
    """Same obsm key name, fewer dimensions — the shape of the reported bug."""
    out = adata.copy()
    out.obsm["embA"] = np.ascontiguousarray(adata.obsm["embA"][:, :d])
    return out


def _make_legacy(cache):
    """Rewrite a 0.3.0 cache in the pre-0.3.0 layout: no width segment in the
    directory name, no n_features/content_sha in the manifest."""
    for d in list(cache.iterdir()):
        legacy = cache / re.sub(r"__d\d+__", "__", d.name)
        d.rename(legacy)
        m = legacy / "manifest.json"
        obj = json.loads(m.read_text())
        obj.pop("content_sha", None)
        obj.pop("n_features", None)
        m.write_text(json.dumps(obj))


# ---------------------------------------------------------------------------
# Change 1 — width in the key
# ---------------------------------------------------------------------------

def test_width_change_does_not_collide(tiny_adata, tmp_path):
    cache = tmp_path / "cache"
    _predict(_fit(tiny_adata, cache), tiny_adata)

    narrow = _narrowed(tiny_adata)
    s4 = _fit(narrow, cache)
    _predict(s4, narrow)          # raised ValueError before the fix

    # embA changed width and must retrain; embB is untouched and still hits
    assert s4.cache_stats_["misses"] == 4
    assert s4.cache_stats_["hits"] == 4
    names = sorted(p.name for p in cache.iterdir() if p.name.startswith("embA__"))
    assert len(names) == 8                     # 4 folds × 2 widths, coexisting
    assert sum("__d8__" in n for n in names) == 4
    assert sum("__d4__" in n for n in names) == 4


def test_manifest_records_width_and_sha(tiny_adata, tmp_path):
    cache = tmp_path / "cache"
    s = _fit(tiny_adata, cache)
    entry = next(p for p in cache.iterdir() if p.name.startswith("embA__"))
    manifest = json.loads((entry / "manifest.json").read_text())
    assert manifest["n_features"] == 8
    assert manifest["content_sha"] == s._emb_sha_["embA"]


# ---------------------------------------------------------------------------
# Change 2 — legacy adoption
# ---------------------------------------------------------------------------

def test_legacy_entries_are_adopted(tiny_adata, tmp_path):
    cache = tmp_path / "cache"
    _fit(tiny_adata, cache)
    _make_legacy(cache)

    s = _fit(tiny_adata, cache)
    assert s.cache_stats_["legacy_adopted"] == 8
    assert s.cache_stats_["hits"] == 8 and s.cache_stats_["misses"] == 0

    links = [p for p in cache.iterdir() if p.is_symlink()]
    assert len(links) == 8 and all(re.search(r"__d8__", p.name) for p in links)

    # the run after adoption resolves the new names directly
    s3 = _fit(tiny_adata, cache)
    assert s3.cache_stats_["hits"] == 8 and s3.cache_stats_["legacy_adopted"] == 0


def test_legacy_not_adopted_at_a_different_width(tiny_adata, tmp_path):
    cache = tmp_path / "cache"
    _fit(tiny_adata, cache)
    _make_legacy(cache)

    s = _fit(_narrowed(tiny_adata), cache)
    # embB is unchanged at 8-d and adopts; embA is now 4-d and must retrain
    assert s.cache_stats_["legacy_adopted"] == 4
    assert s.cache_stats_["misses"] == 4


def test_legacy_cache_ignore_retrains(tiny_adata, tmp_path):
    cache = tmp_path / "cache"
    _fit(tiny_adata, cache)
    _make_legacy(cache)

    s = _fit(tiny_adata, cache, legacy_cache="ignore")
    assert s.cache_stats_["misses"] == 8
    assert s.cache_stats_["legacy_adopted"] == 0


# ---------------------------------------------------------------------------
# Change 3 — advisory content hash
# ---------------------------------------------------------------------------

def test_content_change_warns_then_strict_cache_retrains(tiny_adata, tmp_path, capsys):
    cache = tmp_path / "cache"
    _fit(tiny_adata, cache)

    perturbed = tiny_adata.copy()          # same width, different values
    perturbed.obsm["embA"] = tiny_adata.obsm["embA"] + np.float32(0.5)

    advisory = _fit(perturbed, cache)
    assert "content changed" in capsys.readouterr().out
    assert advisory.cache_stats_["hits"] == 8        # warned, but still used

    strict = _fit(perturbed, cache, strict_cache=True)
    assert strict.cache_stats_["misses"] == 4        # embA retrained
    assert strict.cache_stats_["hits"] == 4          # embB untouched


# ---------------------------------------------------------------------------
# Change 4 — width guard helper
# ---------------------------------------------------------------------------

def test_model_n_features_covers_every_cached_classifier():
    class _Xgb:
        n_features_in_ = 37

    class _Torch:
        D_ = 12

    assert _model_n_features(_Xgb()) == 37
    assert _model_n_features(_Torch()) == 12
    # GlobalProbsWrapper is what a donor fold missing rare cell types produces
    assert _model_n_features(GlobalProbsWrapper(_Xgb(), np.array([0, 1]), 4)) == 37
    assert _model_n_features(GlobalProbsWrapper(_Torch(), np.array([0, 1]), 4)) == 12
    assert _model_n_features(object()) is None       # unknown ⇒ skip the guard
