"""Pretrained reference model (scboost_ensemble.reference_model): fit, save, load and project on CPU.

The synthetic reference is tiny, so the module's sizes (3,000 HVGs, 100 components, scVI 30/128/2) are scaled
down here; the code path is the one the published model runs. Set SCBOOST_TS_V2_MODEL to the downloaded
ts_v2_full directory to also check that the published model loads.
"""

from __future__ import annotations
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

rm = pytest.importorskip("scboost_ensemble.reference_model")


def _counts(rng, n_cells, labels, n_genes=300):
    """Negative-binomial counts with a block of 30 marker genes per class."""
    mu = np.full((n_cells, n_genes), 0.5)
    for k, c in enumerate(sorted(set(labels))):
        mu[np.asarray(labels) == c, 30 * k:30 * (k + 1)] = 4.0
    return rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)


def _adata(rng, donors, n_per=40, classes=("A", "B", "C")):
    import anndata as ad
    import scipy.sparse as sp
    labels = np.tile(np.repeat(classes, n_per), len(donors))
    donor = np.repeat(donors, n_per * len(classes))
    X = sp.csr_matrix(_counts(rng, len(labels), labels))
    obs = pd.DataFrame({"trained_label": labels, "donor_id": donor, "batch": donor},
                       index=[f"{d}_{i}" for i, d in enumerate(donor)])
    var = pd.DataFrame(index=[f"ENSG{j:011d}" for j in range(X.shape[1])])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def small_sizes(monkeypatch):
    monkeypatch.setattr(rm, "N_HVG", 120)
    monkeypatch.setattr(rm, "SCVI_N_HVG", 100)
    monkeypatch.setattr(rm, "N_COMP", 8)
    monkeypatch.setattr(rm, "SCVI_PARAMS", dict(n_latent=4, n_hidden=16, n_layers=1))
    monkeypatch.setattr(rm, "MLP_GENERAL", dict(arch="mlp", hidden=[16], dropout=0.1, weight_decay=1e-4,
                                                lr=1e-2, batch_size=4096, max_epochs=30))


def test_fit_save_load_project(tmp_path, small_sizes):
    for mod in ("scvi", "torch", "skmisc"):
        pytest.importorskip(mod)
    rng = np.random.default_rng(0)
    ref = _adata(rng, [f"r{i}" for i in range(6)])
    query = _adata(rng, ["q0", "q1"], n_per=20)

    m = rm.ReferenceModel(device="cpu", hvg_min_batch=50, metas=("xgboost",), scvi_max_epochs=2)
    m.fit(ref)
    m.set_compartments(pd.Series({"A": "X", "B": "X", "C": "Y"}))
    m.save(tmp_path / "model")

    m2 = rm.ReferenceModel.load(tmp_path / "model", device="cpu", verify=True)
    assert m2.classes == ["A", "B", "C"]
    emb = m.ref_embeddings
    p_fit = rm.stacked_proba(m.stackers["xgboost"], rm.base_view_probs(m.stackers["xgboost"], emb))
    p_load = rm.stacked_proba(m2.stackers["xgboost"], rm.base_view_probs(m2.stackers["xgboost"], emb))
    assert np.array_equal(p_fit, p_load)                      # the saved stacker is the fitted one

    r1 = m2.project(query, donor_key="donor_id", surgery_epochs=2, verbose=False)
    r2 = m2.project(query, donor_key="donor_id", surgery_epochs=2, verbose=False)
    P = r1["proba"]["xgboost"]
    assert P.shape == (query.n_obs, 3) and np.allclose(P.sum(1), 1, atol=1e-4)
    assert set(r1["compartment_label"]["xgboost"]) <= {"X", "Y"}
    assert np.array_equal(P, r2["proba"]["xgboost"])          # seeded surgery: deterministic on CPU
    assert (r1["label"]["xgboost"] == query.obs["trained_label"].values).mean() > 0.8

    # ref_embeddings/ is optional at load time; every other file is required
    import shutil
    shutil.rmtree(tmp_path / "model" / "ref_embeddings")
    rm.ReferenceModel.load(tmp_path / "model", device="cpu", verify=True)
    (tmp_path / "model" / "pca.joblib").unlink()
    with pytest.raises(AssertionError, match="missing files"):
        rm.ReferenceModel.load(tmp_path / "model", device="cpu", verify=True)


def test_legacy_module_names_resolve():
    """Models saved by the manuscript's analysis code pickle RefStacker as lib.reference_model.RefStacker."""
    cls = rm.RefStacker
    with rm._legacy_pickle_names():
        cls.__module__ = "lib.reference_model"
        try:
            blob = pickle.dumps(cls)
        finally:
            cls.__module__ = rm.__name__
    assert b"lib.reference_model" in blob
    assert "lib" not in sys.modules                           # the aliases are removed afterwards
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(blob)
    with rm._legacy_pickle_names():
        assert pickle.loads(blob) is rm.RefStacker


@pytest.mark.skipif(not os.environ.get("SCBOOST_TS_V2_MODEL"), reason="SCBOOST_TS_V2_MODEL not set")
def test_published_model_loads():
    m = rm.ReferenceModel.load(os.environ["SCBOOST_TS_V2_MODEL"], device="cpu", verify=True)
    assert len(m.classes) == 167 and list(m.stackers) == ["xgboost", "torch_logreg_en"]
    st = m.stackers["xgboost"]
    assert st.obsm_keys == list(rm.VIEWS) and st.n_folds == 5
