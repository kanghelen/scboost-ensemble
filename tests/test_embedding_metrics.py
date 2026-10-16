"""Tests for embedding_metrics on synthetic embeddings with known answers."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from scboost_ensemble import embedding_metrics as em


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).normal(size=(2000, 8))


def test_identical_embedding_is_perfectly_similar(X):
    assert em.linear_cka(X, X) == pytest.approx(1.0)
    fit = em.linear_fit(X, X)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["residual_mean"] == pytest.approx(0.0, abs=1e-10)
    assert fit["rank"] == 8
    assert fit["coef"].shape == (8, 8)
    assert np.allclose(em.paired_distance(X, X), 0.0)


def test_cka_and_r2_ignore_rotation_scale_and_shift(X):
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))              # orthogonal
    Y = 3.0 * X @ Q + 5.0
    assert em.linear_cka(X, Y) == pytest.approx(1.0)
    assert em.linear_fit(X, Y)["r2"] == pytest.approx(1.0)
    assert em.linear_fit(Y, X)["r2"] == pytest.approx(1.0)
    # the raw coordinates differ by the rotation, which is exactly why raw distance
    # is not a similarity between different methods
    assert em.paired_distance(X, Y).mean() > 0.1


def test_independent_embeddings_score_near_zero(X):
    Y = np.random.default_rng(2).normal(size=X.shape)
    assert em.linear_cka(X, Y) < 0.05
    assert em.linear_fit(X, Y)["r2"] < 0.05


def test_projection_is_asymmetric(X):
    Y = X[:, :4]                                              # half the dimensions
    assert em.linear_fit(X, Y)["r2"] == pytest.approx(1.0)
    assert em.linear_fit(Y, X)["r2"] == pytest.approx(0.5, abs=0.05)
    assert em.linear_cka(X, Y) == pytest.approx(em.linear_cka(Y, X))
    with pytest.raises(ValueError, match="equal width"):
        em.paired_distance(X, Y)


def test_input_validation(X):
    with pytest.raises(ValueError, match="same cells"):
        em.linear_cka(X, X[:100])
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        em.linear_cka(bad, X)
    assert np.isnan(em.linear_cka(np.ones_like(X), X))


def test_compare_embeddings_table(tiny_adata):
    table = em.compare_embeddings(tiny_adata, verbose=False)
    assert list(table.columns) == ["from", "to", "n_cells", "dim_from", "dim_to", "cka", "r2",
                                   "residual_mean", "residual_median", "residual_max", "rank_from",
                                   "raw_distance_mean", "raw_distance_median", "raw_distance_max"]
    assert set(zip(table["from"], table["to"])) == {("embA", "embB"), ("embB", "embA")}
    assert (table["n_cells"] == tiny_adata.n_obs).all()
    t = table.set_index(["from", "to"])
    ab, ba = t.loc[("embA", "embB")], t.loc[("embB", "embA")]
    assert ab["cka"] == pytest.approx(ba["cka"])
    assert ab["raw_distance_mean"] == pytest.approx(ba["raw_distance_mean"])
    assert 0.0 <= ab["cka"] <= 1.0 and ab["r2"] <= 1.0
    assert table["raw_distance_mean"].notna().all()           # both 8-d

    sub = em.compare_embeddings(tiny_adata, max_cells=100, random_state=0, verbose=False)
    assert (sub["n_cells"] == 100).all()
    with pytest.raises(KeyError, match="not found"):
        em.compare_embeddings(tiny_adata, keys=["embA", "nope"], verbose=False)
    with pytest.raises(ValueError, match="at least two"):
        em.compare_embeddings(tiny_adata, keys=["embA"], verbose=False)


def test_compare_embeddings_mapping_and_unequal_width(X):
    noise = np.random.default_rng(3).normal(size=X.shape)
    table = em.compare_embeddings({"full": X, "half": X[:, :4], "noise": noise}, verbose=False)
    assert len(table) == 6
    t = table.set_index(["from", "to"])
    assert np.isnan(t.loc[("full", "half"), "raw_distance_mean"])
    assert t.loc[("full", "noise"), "raw_distance_mean"] > 0
    assert t.loc[("full", "half"), "r2"] == pytest.approx(1.0)
    assert t.loc[("half", "full"), "r2"] == pytest.approx(0.5, abs=0.05)


def test_similarity_matrix_and_heatmap(X):
    Q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(8, 8)))
    table = em.compare_embeddings({"a": X, "b": X[:, :4], "c": X @ Q}, verbose=False)
    m = em.similarity_matrix(table, "cka")
    assert m.shape == (3, 3) and list(m.index) == ["a", "b", "c"] and list(m.columns) == ["a", "b", "c"]
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m.values, m.values.T)
    assert m.loc["a", "c"] == pytest.approx(1.0)
    r2 = em.similarity_matrix(table, "r2")
    assert r2.loc["a", "b"] == pytest.approx(1.0) and r2.loc["b", "a"] < 0.6   # asymmetric
    assert np.allclose(np.diag(em.similarity_matrix(table, "residual_mean")), 0.0)
    with pytest.raises(KeyError):
        em.similarity_matrix(table, "not_a_metric")

    ax = em.plot_similarity_heatmap(m, title="cka")
    assert isinstance(ax, plt.Axes)
    assert len(ax.texts) == 9                                 # every cell annotated
    plt.close(ax.figure)
    ax2 = em.plot_similarity_heatmap(em.similarity_matrix(table, "raw_distance_mean"))
    assert len(ax2.texts) < 9                                 # NaN cells (unequal width) left blank
    plt.close(ax2.figure)


def test_cli_writes_table_and_heatmap(tiny_adata, tmp_path):
    h5 = tmp_path / "tiny.h5ad"
    tiny_adata.write_h5ad(h5)
    out, fig = tmp_path / "pairs.tsv", tmp_path / "cka.png"
    em.main([str(h5), "--obsm_keys", "embA", "embB", "--out", str(out), "--heatmap", str(fig),
             "--max_cells", "200"])
    table = pd.read_csv(out, sep="\t")
    assert len(table) == 2 and (table["n_cells"] == 200).all()
    assert fig.exists() and fig.stat().st_size > 0
    with pytest.raises(KeyError, match="not in"):
        em.main([str(h5), "--obsm_keys", "embA", "missing", "--out", str(out)])
