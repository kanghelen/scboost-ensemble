"""Pretrained scBoost reference model: fit once on a labelled reference, save every fitted object, reload, and
project a query without refitting anything on it.

Views (the 3 projectable ones, obsm order = canonical alphabetical): NMF, Python_PCA, scVI.
  * Python_PCA: sklearn PCA, 100 components, on log1p(normalize_total 1e4) of 3,000 reference HVGs.
  * NMF: 100 components on the same matrix; the gene basis H (components_) is kept, and transform()
    solves the non-negative least squares for W with H fixed. Backend "sklearn" (sklearn.decomposition.NMF,
    solver "cd") or "torch" (TorchNMF below: the same coordinate-descent updates, run on the GPU).
  * scVI: n_latent 30, n_hidden 128, n_layers 2, 2,000 reference HVGs, counts, batch = donor|assay. A query is
    placed by scArches surgery: its donors become new batch categories; max_epochs 100, weight_decay 0.
HVGs: seurat_v3, batch-aware on the reference only, batches below `hvg_min_batch` cells excluded.
Stacker: StackingEnsembleDonorMetaCached with the MLP "GENERAL" base learner, metas xgboost and torch_logreg_en
(the second fit reuses the base cache), use_batch_feature=False, and the hybrid donor folds of
scboost_ensemble.folds grouped on donor_id.
Two-level output: compartment posterior = sum of the fine posteriors of the classes in each compartment.

Provenance: added in 0.3.0 from the scBoost manuscript's analysis code (lib/reference_model.py, md5
3da9045b058a22b720d1c5cff668a874). The changes are the package-relative imports, the package file md5s in
_versions(), and the loader for models saved by that code (their pickles name the classes `lib.reference_model`,
`lib.folds`, `lib.balanced`), and loading torch tensors onto the CPU first so a model saved on a GPU loads on a
machine without one. Fitting and projection are unchanged.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import platform
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import __version__ as _PKG_VERSION
from .stacking import StackingEnsembleDonorMetaCached
from ._xgb_utils import GlobalProbsWrapper
from .torch_base import TorchBaseClassifier
from .torch_meta import TorchElasticNetMeta
from .folds import HybridFoldsMixin
from .balanced import BalancedLossMixin

PKG_DIR = Path(__file__).resolve().parent

VIEWS = ("NMF", "Python_PCA", "scVI")
N_COMP = 100
N_HVG = 3000
SCVI_N_HVG = 2000
SCVI_PARAMS = dict(n_latent=30, n_hidden=128, n_layers=2)
SURGERY = dict(max_epochs=100, plan_kwargs=dict(weight_decay=0.0))
MLP_GENERAL = dict(arch="mlp", hidden=[512, 256], dropout=0.50, weight_decay=1e-4, lr=1.5e-3, batch_size=4096)
METAS = ("xgboost", "torch_logreg_en")
TARGET_SUM = 1e4
SEED = 42
MAX_MISSING = 0.10
QUERY_BATCH_PREFIX = "query|"


class RefStacker(HybridFoldsMixin, StackingEnsembleDonorMetaCached):
    """The stacker with the hybrid donor folds (scboost_ensemble.folds); pass batch_key="donor_id"."""


class RefStackerBalanced(BalancedLossMixin, HybridFoldsMixin, StackingEnsembleDonorMetaCached):
    """RefStacker with class-balanced losses (scboost_ensemble.balanced); pass class_weight_beta."""


# Models saved by the manuscript's analysis code pickle these classes under the module names on the left.
_LEGACY_MODULES = {"lib.reference_model": __name__, "lib.folds": f"{__package__}.folds",
                   "lib.balanced": f"{__package__}.balanced"}


@contextlib.contextmanager
def _legacy_pickle_names():
    """While loading, resolve the legacy `lib.*` module names to this package's modules. Names already
    importable in this process are left alone; the aliases are removed afterwards."""
    added = []
    if "lib" not in sys.modules:
        sys.modules["lib"] = types.ModuleType("lib")
        added.append("lib")
    for old, new in _LEGACY_MODULES.items():
        if old not in sys.modules:
            sys.modules[old] = importlib.import_module(new)
            added.append(old)
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


@contextlib.contextmanager
def _torch_storages_on_cpu():
    """While loading, place pickled torch tensors on the CPU. A meta-learner fitted on a GPU pickles CUDA tensors,
    which torch cannot restore on a machine without one; set_stacker_device then moves them to the requested device,
    so the values are unchanged."""
    try:
        import io
        import torch
        import torch.storage
    except ImportError:          # no torch: nothing pickled by torch can be in the file
        yield
        return
    orig = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    try:
        yield
    finally:
        torch.storage._load_from_bytes = orig


# ---------------------------------------------------------------------------------------------------------
# I/O and normalisation
# ---------------------------------------------------------------------------------------------------------

def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_LIB_MD5_AT_IMPORT = md5(Path(__file__))      # the code that actually ran, even if the file changes later


def read_rows_csr(path, rows, key="X", block=50_000, sparse_frac=0.02):
    """Rows (sorted, unique) of a CSR matrix stored in an h5ad (`key` = "X" or "layers/counts"), as float32 CSR
    with int32 indices. Dense requests read contiguous row blocks; sparse requests read only the runs needed."""
    import h5py
    rows = np.asarray(rows, dtype=np.int64)
    assert rows.ndim == 1 and (np.diff(rows) > 0).all(), "rows must be sorted and unique"
    with h5py.File(path, "r") as f:
        g = f[key]
        assert g.attrs.get("encoding-type") == "csr_matrix", dict(g.attrs)
        n, G = (int(x) for x in g.attrs["shape"])
        indptr = g["indptr"][:]
        D, I, P = g["data"], g["indices"], [np.zeros(1, dtype=np.int64)]
        datas, inds, total = [], [], 0
        for b0 in range(0, n, block):
            b1 = min(n, b0 + block)
            lo, hi = np.searchsorted(rows, [b0, b1])
            want = rows[lo:hi]
            if len(want) == 0:
                continue
            if len(want) / (b1 - b0) >= sparse_frac:
                s, e = int(indptr[b0]), int(indptr[b1])
                d, ix = D[s:e], I[s:e]
                ip = indptr[b0:b1 + 1] - s
                for r in want - b0:
                    a, z = ip[r], ip[r + 1]
                    datas.append(d[a:z]); inds.append(ix[a:z].astype(np.int32))
                    total += z - a; P.append(np.array([total]))
            else:
                for r in want:
                    a, z = int(indptr[r]), int(indptr[r + 1])
                    datas.append(D[a:z]); inds.append(I[a:z].astype(np.int32))
                    total += z - a; P.append(np.array([total]))
    data = np.concatenate(datas).astype(np.float32, copy=False) if datas else np.zeros(0, np.float32)
    ind = np.concatenate(inds) if inds else np.zeros(0, np.int32)
    ptr = np.concatenate(P).astype(np.int64)
    return sp.csr_matrix((data, ind, ptr), shape=(len(rows), G))


def log_normalize(counts, totals, target=TARGET_SUM):
    """log1p(counts / totals * target), CSR float32; totals are computed beforehand over the size-factor genes."""
    totals = np.asarray(totals, dtype=np.float64).ravel()
    s = np.divide(target, totals, out=np.zeros_like(totals), where=totals > 0).astype(np.float32)
    X = sp.csr_matrix(counts, dtype=np.float32, copy=True)
    X.data *= np.repeat(s, np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    return X


def strip_version(ids):
    return np.array([re.sub(r"\.\d+$", "", str(g)) for g in ids])


# ---------------------------------------------------------------------------------------------------------
# NMF on the GPU (the same coordinate-descent updates as sklearn's solver "cd", shuffle=False)
# ---------------------------------------------------------------------------------------------------------

class TorchNMF:
    """Frobenius NMF without regularisation, X ~ W H, fitted by coordinate descent (HALS) on the GPU.

    Mirrors sklearn.decomposition.NMF(solver="cd", shuffle=False, alpha_W=0, alpha_H=0): the same nndsvda
    initialisation (sklearn's own `_initialize_nmf`), the same column-wise updates
    W[:, t] <- max(0, W[:, t] - (W HH^T[:, t] - XH^T[:, t]) / HH^T[t, t]), the same projected-gradient
    violation and the same stopping rule (violation / violation_at_iteration_1 <= tol). transform(X) holds
    components_ fixed and starts W at zeros, as sklearn's transform does for solver "cd".
    Arithmetic is float32 on the GPU (sklearn's Cython kernel also runs in float32 for float32 input).
    """

    def __init__(self, n_components=N_COMP, init="nndsvda", max_iter=500, tol=1e-4, random_state=SEED,
                 device="cuda:0", row_block=262_144):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.device = device
        self.row_block = row_block

    @staticmethod
    def _cd_update(W, HHt, XHt):
        """One sweep over the k columns of W (in place). Returns the summed |projected gradient|."""
        import torch
        viol = torch.zeros((), dtype=torch.float64, device=W.device)
        for t in range(W.shape[1]):
            grad = W @ HHt[:, t] - XHt[:, t]
            pg = torch.where(W[:, t] == 0, torch.clamp(grad, max=0.0), grad)
            viol += pg.abs().sum(dtype=torch.float64)
            hess = HHt[t, t]
            if hess != 0:
                W[:, t] = torch.clamp(W[:, t] - grad / hess, min=0.0)
        return float(viol)

    def _to_dev(self, X):
        import torch
        if sp.issparse(X):
            X = X.toarray()
        return torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=self.device)

    def fit_transform(self, X):
        import torch
        from sklearn.decomposition._nmf import _initialize_nmf
        X = X.astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
        W0, H0 = _initialize_nmf(X, self.n_components, init=self.init, random_state=self.random_state)
        Xt = self._to_dev(X)
        W = torch.as_tensor(np.asarray(W0, dtype=np.float32), device=self.device).contiguous()
        Ht = torch.as_tensor(np.asarray(H0, dtype=np.float32).T.copy(), device=self.device).contiguous()
        v0, it, self.violations_ = None, 0, []
        for it in range(1, self.max_iter + 1):
            v = self._cd_update(W, Ht.T @ Ht, Xt @ Ht)
            v += self._cd_update(Ht, W.T @ W, Xt.T @ W)
            self.violations_.append(v)
            if it == 1:
                v0 = v
            if v0 == 0 or v / v0 <= self.tol:
                break
        self.n_iter_ = it
        self.components_ = Ht.T.contiguous().cpu().numpy()
        self.reconstruction_err_ = float(torch.linalg.norm(Xt - W @ Ht.T))
        del Xt
        torch.cuda.empty_cache()
        return W.cpu().numpy()

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        """Rows are independent given H, but the stopping rule is global (as in sklearn); X is processed whole
        when it fits, otherwise in row blocks with the rule applied per block (recorded in n_iter_transform_)."""
        import torch
        Ht = torch.as_tensor(np.ascontiguousarray(self.components_.T, dtype=np.float32), device=self.device)
        HHt = Ht.T @ Ht
        n = X.shape[0]
        out, iters = np.zeros((n, self.n_components), dtype=np.float32), []
        for a in range(0, n, self.row_block):
            Xb = self._to_dev(X[a:a + self.row_block])
            XHt = Xb @ Ht
            # sklearn (solver "cd", update_H=False) starts W at zeros (_nmf.py _check_w_h)
            W = torch.zeros((Xb.shape[0], self.n_components), dtype=torch.float32, device=self.device)
            v0 = None
            for it in range(1, self.max_iter + 1):
                v = self._cd_update(W, HHt, XHt)
                if it == 1:
                    v0 = v
                if v0 == 0 or v / v0 <= self.tol:
                    break
            iters.append(it)
            out[a:a + self.row_block] = W.cpu().numpy()
            del Xb, XHt, W
        self.n_iter_transform_ = iters
        return out

    def get_state(self):
        return dict(n_components=self.n_components, init=self.init, max_iter=self.max_iter, tol=self.tol,
                    random_state=self.random_state, components_=self.components_, n_iter_=self.n_iter_,
                    reconstruction_err_=self.reconstruction_err_)

    @classmethod
    def from_state(cls, state, device="cuda:0"):
        m = cls(state["n_components"], state["init"], state["max_iter"], state["tol"], state["random_state"],
                device=device)
        m.components_ = state["components_"]
        m.n_iter_ = state["n_iter_"]
        m.reconstruction_err_ = state["reconstruction_err_"]
        return m


# ---------------------------------------------------------------------------------------------------------
# Helpers for the stacker
# ---------------------------------------------------------------------------------------------------------

def _inner(clf):
    return clf.clf if isinstance(clf, GlobalProbsWrapper) else clf


def set_stacker_device(stacker, device):
    """Point every fitted base and meta classifier at one device ("cpu" or "cuda:N") for inference."""
    import torch
    for key in stacker.obsm_keys:
        for clf in stacker.base_classifiers_[key]:
            c = _inner(clf)
            if isinstance(c, TorchBaseClassifier):
                c.device = device
                c._module = None          # rebuilt from the cpu state_dict on the next predict
    import pickle
    for k, m in enumerate(stacker.meta_classifiers_):
        c = _inner(m)
        if isinstance(c, TorchElasticNetMeta):
            c.device = device
            for a in ("_W", "_b", "_mean_", "_std_"):
                if hasattr(c, a):
                    setattr(c, a, getattr(c, a).to(torch.device(device)))
        elif hasattr(c, "set_params") and hasattr(c, "get_xgb_params"):
            # A booster that has predicted on the GPU keeps GPU-style predictions after set_params(device="cpu")
            # (|dp| ~1e-7, measured with xgboost 2.1.4). A pickle round trip gives a fresh booster.
            c = pickle.loads(pickle.dumps(c))
            c.set_params(device=device)
            if isinstance(m, GlobalProbsWrapper):
                m.clf = c
            else:
                stacker.meta_classifiers_[k] = c
    stacker.device = device


def base_view_probs(stacker, embs):
    """{view: [5 arrays (n, C)]} from the fold-k base classifiers, as the engine's _test_meta_for_fold does."""
    out = {}
    for key in stacker.obsm_keys:
        out[key] = [stacker.base_classifiers_[key][k].predict_proba(embs[key]).astype(np.float32)
                    for k in range(stacker.n_folds)]
    return out


def stacked_proba(stacker, view_probs):
    """The engine's predict_proba (stacking.py:533-545) from precomputed base probabilities."""
    C = stacker.n_cell_types
    probas = []
    for k in range(stacker.n_folds):
        n = next(iter(view_probs.values()))[k].shape[0]
        X = np.zeros((n, len(stacker.obsm_keys) * C), dtype=np.float32)
        for ki, key in enumerate(stacker.obsm_keys):
            X[:, ki * C:(ki + 1) * C] = view_probs[key][k]
        probas.append(stacker.meta_classifiers_[k].predict_proba(X))
    return np.mean(probas, axis=0)


def _versions():
    import anndata, scanpy, scipy, sklearn, torch, xgboost, scvi
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__, torch=torch.__version__,
                scvi_tools=scvi.__version__, scanpy=scanpy.__version__, anndata=anndata.__version__,
                xgboost=xgboost.__version__, host=platform.node(), scboost_ensemble=_PKG_VERSION,
                engine_stacking_md5=md5(PKG_DIR / "stacking.py"),
                lib_reference_model_md5=_LIB_MD5_AT_IMPORT, lib_folds_md5=md5(PKG_DIR / "folds.py"),
                lib_balanced_md5=md5(PKG_DIR / "balanced.py"))


# ---------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------

class ReferenceModel:
    def __init__(self, nmf_backend="sklearn", hvg_min_batch=2000, device="cuda:0", n_gpus=3,
                 base_cache_dir=None, metas=METAS, seed=SEED, nmf_max_iter=500, scvi_max_epochs=None,
                 notes=None, class_weight_beta=None):
        """class_weight_beta: None = the engine's unweighted losses (every model before 2026-10-27); a number =
        class-balanced losses in all base and meta learners, w_c ∝ n_c^-beta (scboost_ensemble.balanced)."""
        assert nmf_backend in ("sklearn", "torch")
        self.class_weight_beta = None if class_weight_beta is None else float(class_weight_beta)
        self.nmf_backend = nmf_backend
        self.hvg_min_batch = hvg_min_batch
        self.device = device
        self.n_gpus = n_gpus
        self.base_cache_dir = base_cache_dir
        self.metas = tuple(metas)
        self.seed = seed
        self.nmf_max_iter = nmf_max_iter
        self.scvi_max_epochs = scvi_max_epochs
        self.notes = dict(notes or {})
        self.timings = {}

    # ----------------------------------------------------------------------------------------------- fit
    def _select_hvgs(self, counts, genes, batch, n_top):
        import anndata as ad
        import scanpy as sc
        batch = np.asarray(batch)
        vc = pd.Series(batch).value_counts()
        big = list(vc[vc >= self.hvg_min_batch].index)
        # seurat_v3's per-batch loess can be singular (skmisc: "reciprocal condition number"); scanpy then fails
        # for all batches. Test each batch alone once and exclude those whose loess fails.
        if not hasattr(self, "_hvg_loess_failed"):
            failed = []
            for b in big:
                sb = ad.AnnData(X=counts[np.where(batch == b)[0]], var=pd.DataFrame(index=genes))
                try:
                    sc.pp.highly_variable_genes(sb, flavor="seurat_v3", n_top_genes=n_top)
                except ValueError as e:
                    print(f"HVG: batch {b} ({vc[b]} cells): loess failed ({e}); excluded")
                    failed.append(b)
            self._hvg_loess_failed = failed
        big = [b for b in big if b not in self._hvg_loess_failed]
        mask = np.isin(batch, big)
        dropped = sorted(set(vc.index) - set(big))
        sub = ad.AnnData(X=counts[mask] if not mask.all() else counts,
                         obs=pd.DataFrame({"b": np.asarray(batch)[mask]}, index=np.arange(mask.sum()).astype(str)),
                         var=pd.DataFrame(index=genes))
        how = "seurat_v3, batch-aware"
        try:
            sc.pp.highly_variable_genes(sub, flavor="seurat_v3", n_top_genes=n_top, batch_key="b")
        except ValueError as e:
            print(f"HVG: batch-aware seurat_v3 failed ({e}); pooling across batches")
            sc.pp.highly_variable_genes(sub, flavor="seurat_v3", n_top_genes=n_top)
            how = "seurat_v3, pooled (batch-aware failed)"
        hv = sub.var.highly_variable.values
        info = dict(n=int(hv.sum()), method=how, batches_used=int(len(big)), batches_excluded=dropped,
                    batches_excluded_loess=list(self._hvg_loess_failed), cells_used=int(mask.sum()),
                    min_batch=self.hvg_min_batch)
        print(f"HVG {n_top}: {info['n']} genes from {info['batches_used']} batches ({how}); excluded {len(dropped)} batches")
        return list(np.asarray(genes)[hv]), info

    def fit(self, adata, label_key="trained_label", donor_key="donor_id", batch_key="batch",
            expected_folds=None, ref_meta=None, stage_dir=None, views_only=False):
        """adata: reference AnnData; X = raw counts (CSR), var_names = Ensembl IDs, obs has label/donor/batch.

        views_only=True stops after the 3 views (used to co-embed a reference and a query):
        HVGs, Python_PCA, NMF and scVI are fitted on every cell of `adata` (reference + query) with the same recipe,
        and no stacker is fitted; label_key is then not used by any fitted object."""
        import anndata as ad
        import torch
        T = self.timings
        t_all = time.time()
        X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        X = X.tocsr()
        self.genes = list(map(str, adata.var_names))
        assert len(set(self.genes)) == len(self.genes), "duplicate gene IDs"
        labels = adata.obs[label_key].astype(str).values
        donors = adata.obs[donor_key].astype(str).values
        batches = adata.obs[batch_key].astype(str).values
        totals = np.asarray(X.sum(1)).ravel()
        gsum = np.asarray(X.sum(0)).ravel().astype(np.float64)
        self.gene_count_frac = (gsum / gsum.sum()).astype(np.float64)
        self.n_ref = adata.n_obs
        self.ref_obs_names = np.asarray(adata.obs_names).astype(str)
        self.ref_meta = dict(ref_meta or {})
        self.ref_meta.update(n_cells=int(adata.n_obs), n_donors=int(len(set(donors))), n_batches=int(len(set(batches))),
                             n_genes=int(len(self.genes)), label_key=label_key, donor_key=donor_key,
                             batch_key=batch_key, zero_total_cells=int((totals == 0).sum()))

        # optional stage directory: each finished view is written there, and a re-run on the same cells reloads it
        stage = Path(stage_dir) if stage_dir else None
        if stage is not None:
            stage.mkdir(parents=True, exist_ok=True)
            cells_sha = hashlib.sha1("\n".join(self.ref_obs_names).encode()).hexdigest()
            ck = stage / "cells.sha1"
            if ck.exists():
                assert ck.read_text().strip() == cells_sha, f"stage {stage} belongs to other cells"
            else:
                ck.write_text(cells_sha + "\n")
        import joblib

        t = time.time()
        if stage is not None and (stage / "hvg.json").exists():
            h = json.loads((stage / "hvg.json").read_text())
            self.hvg_pca, self.hvg_scvi, self.hvg_info = h["pca_nmf"], h["scvi"], h["info"]
            T["hvg_s"] = h["seconds"]
            print("HVG: reloaded from stage")
        else:
            self.hvg_pca, info3 = self._select_hvgs(X, self.genes, batches, N_HVG)
            self.hvg_scvi, info2 = self._select_hvgs(X, self.genes, batches, SCVI_N_HVG)
            self.hvg_info = {"pca_nmf": info3, "scvi": info2}
            T["hvg_s"] = round(time.time() - t, 1)
            if stage is not None:
                (stage / "hvg.json").write_text(json.dumps(dict(pca_nmf=self.hvg_pca, scvi=self.hvg_scvi,
                                                                info=self.hvg_info, seconds=T["hvg_s"])))
        gidx = {g: i for i, g in enumerate(self.genes)}
        i3 = np.array([gidx[g] for g in self.hvg_pca])
        i2 = np.array([gidx[g] for g in self.hvg_scvi])

        # Python_PCA and NMF on log1p(normalize_total 1e4), size factors over all reference genes
        Z = {}
        done = lambda name: stage is not None and (stage / f"{name}.joblib").exists() and (stage / f"Z_{name}.npy").exists()
        need_x = not (done("Python_PCA") and done("NMF"))
        if need_x:
            t = time.time()
            Xn = log_normalize(X[:, i3], totals)
            Xd = Xn.toarray()
            T["lognorm_s"] = round(time.time() - t, 1)
        from sklearn.decomposition import PCA, NMF
        if done("Python_PCA"):
            self.pca = joblib.load(stage / "Python_PCA.joblib")
            Z["Python_PCA"] = np.load(stage / "Z_Python_PCA.npy")
            T.update(json.loads((stage / "Python_PCA.json").read_text()))
            print("PCA: reloaded from stage")
        else:
            t = time.time()
            self.pca = PCA(n_components=N_COMP, random_state=self.seed).fit(Xd)
            Z["Python_PCA"] = self.pca.transform(Xd).astype(np.float32)
            T["pca_fit_transform_s"] = round(time.time() - t, 1)
            if stage is not None:
                joblib.dump(self.pca, stage / "Python_PCA.joblib")
                np.save(stage / "Z_Python_PCA.npy", Z["Python_PCA"])
                (stage / "Python_PCA.json").write_text(json.dumps({"pca_fit_transform_s": T["pca_fit_transform_s"]}))
        print(f"PCA {T['pca_fit_transform_s']}s; explained variance (first 5) {np.round(self.pca.explained_variance_ratio_[:5], 4)}")

        if done("NMF"):
            nm = joblib.load(stage / "NMF.joblib")
            self.nmf = nm if self.nmf_backend == "sklearn" else TorchNMF.from_state(nm, device=self.device)
            Z["NMF"] = np.load(stage / "Z_NMF.npy")
            T.update(json.loads((stage / "NMF.json").read_text()))
            print("NMF: reloaded from stage")
        else:
            t = time.time()
            if self.nmf_backend == "sklearn":
                self.nmf = NMF(n_components=N_COMP, init="nndsvda", max_iter=self.nmf_max_iter, random_state=self.seed)
                self.nmf.fit(Xn)
                T["nmf_fit_s"] = round(time.time() - t, 1)
                t = time.time()
                Z["NMF"] = self.nmf.transform(Xn).astype(np.float32)
            else:
                self.nmf = TorchNMF(n_components=N_COMP, max_iter=self.nmf_max_iter, random_state=self.seed,
                                    device=self.device)
                self.nmf.fit(Xd)
                T["nmf_fit_s"] = round(time.time() - t, 1)
                t = time.time()
                Z["NMF"] = self.nmf.transform(Xd).astype(np.float32)
            T["nmf_transform_s"] = round(time.time() - t, 1)
            if stage is not None:
                joblib.dump(self.nmf if self.nmf_backend == "sklearn" else self.nmf.get_state(), stage / "NMF.joblib")
                np.save(stage / "Z_NMF.npy", Z["NMF"])
                (stage / "NMF.json").write_text(json.dumps({k: T[k] for k in ("nmf_fit_s", "nmf_transform_s")}))
        print(f"NMF ({self.nmf_backend}) fit {T['nmf_fit_s']}s, n_iter {self.nmf.n_iter_}, transform {T['nmf_transform_s']}s")
        if need_x:
            del Xd, Xn

        # scVI on counts of 2,000 HVGs, batch = donor|assay
        import scvi
        t = time.time()
        scvi.settings.seed = self.seed
        a2 = ad.AnnData(X=X[:, i2].astype(np.float32), obs=pd.DataFrame({"scvi_batch": batches}, index=self.ref_obs_names),
                        var=pd.DataFrame(index=self.hvg_scvi))
        dev_idx = int(self.device.split(":")[1]) if self.device.startswith("cuda") else None
        tk = dict(accelerator="gpu", devices=[dev_idx]) if dev_idx is not None else dict(accelerator="cpu")
        if stage is not None and (stage / "scvi" / "model.pt").exists() and (stage / "Z_scVI.npy").exists():
            self.scvi_model = scvi.model.SCVI.load(str(stage / "scvi"), adata=a2,
                                                   **({"accelerator": "gpu", "device": [dev_idx]} if dev_idx is not None else {"accelerator": "cpu"}))
            Z["scVI"] = np.load(stage / "Z_scVI.npy")
            h = json.loads((stage / "scVI.json").read_text())
            self.scvi_epochs, T["scvi_fit_s"] = h["epochs"], h["scvi_fit_s"]
            print("scVI: reloaded from stage")
        else:
            scvi.model.SCVI.setup_anndata(a2, batch_key="scvi_batch")
            self.scvi_model = scvi.model.SCVI(a2, **SCVI_PARAMS)
            self.scvi_model.train(max_epochs=self.scvi_max_epochs, **tk)
            Z["scVI"] = self.scvi_model.get_latent_representation().astype(np.float32)
            self.scvi_epochs = int(self.scvi_model.history["elbo_train"].shape[0])
            T["scvi_fit_s"] = round(time.time() - t, 1)
            if stage is not None:
                self.scvi_model.save(str(stage / "scvi"), overwrite=True, save_anndata=False)
                np.save(stage / "Z_scVI.npy", Z["scVI"])
                (stage / "scVI.json").write_text(json.dumps(dict(epochs=self.scvi_epochs, scvi_fit_s=T["scvi_fit_s"])))
        print(f"scVI {T['scvi_fit_s']}s, {self.scvi_epochs} epochs")
        for v in VIEWS:
            assert np.isfinite(Z[v]).all(), f"{v} has non-finite values"
        self.ref_embeddings = Z
        if views_only:
            T["fit_total_s"] = round(time.time() - t_all, 1)
            return self

        # stacker(s)
        A = ad.AnnData(X=sp.csr_matrix((adata.n_obs, 1), dtype=np.float32),
                       obs=pd.DataFrame({"trained_label": labels, "donor_id": donors}, index=self.ref_obs_names))
        for v in VIEWS:
            A.obsm[v] = Z[v]
        self.stackers = {}
        self.fit_info = {}
        for meta in self.metas:
            t = time.time()
            kw = dict(obsm_keys=list(VIEWS), label_key="trained_label", batch_key="donor_id",
                      use_batch_feature=False, n_folds=5, cv_scheme="donor", device=self.device,
                      n_gpus=self.n_gpus, algo_params=dict(MLP_GENERAL), meta_algo=meta, base_algo="mlp",
                      seed=self.seed, base_cache_dir=self.base_cache_dir)
            if self.class_weight_beta is None:
                st = RefStacker(**kw)
            else:
                st = RefStackerBalanced(class_weight_beta=self.class_weight_beta, **kw)
            st.fit(A)
            sizes = np.bincount(st.fold_assignments_, minlength=st.n_folds)
            assert (sizes > 0).all(), f"empty CV fold: {sizes.tolist()}"
            if expected_folds is not None:
                assert (st.fold_assignments_ == np.asarray(expected_folds)).all(), "folds differ from the cell list"
            self.stackers[meta] = st
            T[f"stacker_{meta}_s"] = round(time.time() - t, 1)
            self.fit_info[meta] = dict(fold_sizes=sizes.tolist(), cache_stats=dict(st.cache_stats_),
                                       importance=st.get_method_importance(),
                                       summary=st.get_training_summary())
            print(f"stacker {meta}: {T[f'stacker_{meta}_s']}s, folds {sizes.tolist()}, cache {st.cache_stats_}")
        st0 = self.stackers[self.metas[0]]
        self.classes = list(map(str, st0.label_encoder_.classes_))
        for st in self.stackers.values():
            assert list(map(str, st.label_encoder_.classes_)) == self.classes
        ft = st0.fold_table_.copy()          # indexed by the encoded label (as str); name the classes
        ft.index = [self.classes[int(i)] for i in ft.index]
        ft.index.name = "trained_label"
        self.fold_table = ft
        T["fit_total_s"] = round(time.time() - t_all, 1)
        return self

    def set_compartments(self, compartment_map):
        """compartment_map: Series trained_label -> compartment; every trained class must have one."""
        cm = pd.Series(compartment_map).astype(str)
        miss = [c for c in self.classes if c not in cm.index]
        assert not miss, f"classes without a compartment: {miss}"
        self.compartment_of = {c: cm[c] for c in self.classes}
        self.compartments = sorted(set(self.compartment_of.values()))
        return self

    # ----------------------------------------------------------------------------------------- save/load
    def save(self, out_dir, extra_files=None):
        import joblib
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        joblib.dump(self.pca, out / "pca.joblib")
        if self.nmf_backend == "sklearn":
            joblib.dump(self.nmf, out / "nmf.joblib")
        else:
            joblib.dump(self.nmf.get_state(), out / "nmf.joblib")
        self.scvi_model.save(str(out / "scvi"), overwrite=True, save_anndata=False)
        for meta, st in self.stackers.items():
            joblib.dump(st, out / f"stacker_{meta}.joblib", compress=3)
        np.save(out / "gene_count_frac.npy", self.gene_count_frac)
        emb = out / "ref_embeddings"
        emb.mkdir(exist_ok=True)
        for v in VIEWS:
            np.save(emb / f"{v}.npy", self.ref_embeddings[v])
        pd.DataFrame({"obs_name": self.ref_obs_names,
                      "fold": self.stackers[self.metas[0]].fold_assignments_}).to_parquet(out / "train_cells.parquet")
        label_map = dict(classes=self.classes, compartment_of=self.compartment_of, compartments=self.compartments)
        (out / "label_map.json").write_text(json.dumps(label_map, indent=1))
        self.fold_table.to_csv(out / "fold_table.tsv", sep="\t")
        for name, src in (extra_files or {}).items():
            (out / name).write_bytes(Path(src).read_bytes())
        T = self.timings
        T["save_s"] = round(time.time() - t, 1)
        manifest = dict(
            created=time.strftime("%Y-%m-%d %H:%M:%S"), views=list(VIEWS), metas=list(self.metas),
            genes=self.genes, hvg_pca_nmf=self.hvg_pca, hvg_scvi=self.hvg_scvi, hvg_info=self.hvg_info,
            normalisation=dict(rule="log1p(normalize_total(1e4)); size factor = total counts over all reference "
                                    "genes; for a query, over the reference genes present in the query",
                               target_sum=TARGET_SUM),
            params=dict(n_comp=N_COMP, n_hvg=N_HVG, scvi_n_hvg=SCVI_N_HVG, scvi=SCVI_PARAMS,
                        scvi_max_epochs=self.scvi_max_epochs, scvi_epochs_run=self.scvi_epochs,
                        surgery=SURGERY, mlp=MLP_GENERAL, nmf_backend=self.nmf_backend,
                        nmf=dict(init="nndsvda", max_iter=self.nmf_max_iter, n_iter=int(self.nmf.n_iter_)),
                        hvg_min_batch=self.hvg_min_batch, seed=self.seed, max_missing=MAX_MISSING,
                        stacker=dict(batch_key="donor_id", use_batch_feature=False, n_folds=5,
                                     folds="scboost_ensemble.folds hybrid",
                                     base_cache_dir=str(self.base_cache_dir)),
                        class_weight_beta=self.class_weight_beta),
            classes=self.classes, n_classes=len(self.classes), compartments=self.compartments,
            reference=self.ref_meta, fit_info=self.fit_info, timings=self.timings, notes=self.notes,
            versions=_versions(),
        )
        files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("manifest.json", "reload_gate.json"))
        manifest["files_md5"] = {str(p.relative_to(out)): md5(p) for p in files}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        print(f"saved {out} ({len(files)} files, {T['save_s']}s)")
        return out

    @classmethod
    def load(cls, model_dir, device="cuda:0", verify=True):
        """verify=True checks the md5 of every file in the manifest. The reference embeddings (ref_embeddings/)
        are not needed to project a query, so they may be absent; every other file must be present."""
        import joblib
        import scvi
        d = Path(model_dir)
        man = json.loads((d / "manifest.json").read_text())
        if verify:
            optional = lambda f: f.startswith("ref_embeddings/")
            missing = [f for f in man["files_md5"] if not (d / f).exists()]
            assert all(optional(f) for f in missing), f"missing files: {[f for f in missing if not optional(f)]}"
            bad = [f for f, h in man["files_md5"].items() if f not in missing and md5(d / f) != h]
            assert not bad, f"md5 mismatch: {bad}"
        p = man["params"]
        m = cls(nmf_backend=p["nmf_backend"], hvg_min_batch=p["hvg_min_batch"], device=device,
                metas=man["metas"], seed=p["seed"], nmf_max_iter=p["nmf"]["max_iter"],
                scvi_max_epochs=p["scvi_max_epochs"], notes=man.get("notes"),
                class_weight_beta=p.get("class_weight_beta"))     # absent in models saved before 2026-10-27 = None
        m.manifest = man
        m.genes, m.hvg_pca, m.hvg_scvi, m.hvg_info = man["genes"], man["hvg_pca_nmf"], man["hvg_scvi"], man["hvg_info"]
        m.scvi_epochs = p["scvi_epochs_run"]
        m.ref_meta, m.fit_info, m.timings = man["reference"], man["fit_info"], dict(man["timings"])
        m.pca = joblib.load(d / "pca.joblib")
        nm = joblib.load(d / "nmf.joblib")
        m.nmf = nm if p["nmf_backend"] == "sklearn" else TorchNMF.from_state(nm, device=device)
        m.scvi_dir = str(d / "scvi")
        m.scvi_model = None
        with _legacy_pickle_names(), _torch_storages_on_cpu():
            m.stackers = {meta: joblib.load(d / f"stacker_{meta}.joblib") for meta in man["metas"]}
        for st in m.stackers.values():
            set_stacker_device(st, device)
        m.gene_count_frac = np.load(d / "gene_count_frac.npy")
        lm = json.loads((d / "label_map.json").read_text())
        m.classes, m.compartment_of, m.compartments = lm["classes"], lm["compartment_of"], lm["compartments"]
        m.model_dir = str(d)
        return m

    def set_device(self, device):
        self.device = device
        for st in self.stackers.values():
            set_stacker_device(st, device)
        if self.nmf_backend == "torch":
            self.nmf.device = device
        return self

    # ------------------------------------------------------------------------------------------- project
    def project(self, query, donor_key, counts_layer=None, gene_id_key=None, allow_missing=False,
                surgery_epochs=SURGERY["max_epochs"], seed=None, return_embeddings=True, verbose=True):
        """Place query cells with the frozen maps and predict. Reads no query label.

        query: AnnData with raw counts in X (counts_layer=None), in layers[counts_layer], or in raw.X
        (counts_layer="raw"); var_names (or var[gene_id_key]) are Ensembl IDs.
        Returns a dict of arrays (see the end of this function)."""
        import anndata as ad
        import scvi
        T, t_all = {}, time.time()
        seed = self.seed if seed is None else seed
        if counts_layer is None:
            C = query.X
        elif counts_layer == "raw":
            C = query.raw.X
        else:
            C = query.layers[counts_layer]
        C = sp.csr_matrix(C, dtype=np.float32)
        qg = strip_version(query.var[gene_id_key].values if gene_id_key else query.var_names)
        if counts_layer == "raw":
            qg = strip_version(query.raw.var_names)
        assert C.shape[1] == len(qg)
        nz = C.data[: min(len(C.data), 1_000_000)]
        noninteger = float(np.mean(nz != np.round(nz))) if len(nz) else 0.0
        qpos = {}
        for j, g in enumerate(qg):
            qpos.setdefault(g, j)
        n_dup = len(qg) - len(qpos)

        # gene matching on Ensembl ID; missing reference genes are zero-filled
        report = dict(query_genes=int(len(qg)), duplicated_query_ids=int(n_dup), noninteger_frac=noninteger)
        for name, lst in (("pca_nmf", self.hvg_pca), ("scvi", self.hvg_scvi)):
            miss = [g for g in lst if g not in qpos]
            report[f"missing_{name}"] = len(miss)
            report[f"missing_frac_{name}"] = len(miss) / len(lst)
        present_ref = np.array([g in qpos for g in self.genes])
        report["ref_genes_present"] = int(present_ref.sum())
        report["size_factor_coverage"] = float(self.gene_count_frac[present_ref].sum())
        worst = max(report["missing_frac_pca_nmf"], report["missing_frac_scvi"])
        if verbose:
            print(f"project: {query.n_obs} cells; missing HVGs {report['missing_frac_pca_nmf']:.3%} (PCA/NMF), "
                  f"{report['missing_frac_scvi']:.3%} (scVI); size-factor coverage {report['size_factor_coverage']:.4f}")
        if worst > MAX_MISSING and not allow_missing:
            raise ValueError(f"query lacks {worst:.1%} of a reference HVG list (> {MAX_MISSING:.0%}); "
                             f"pass allow_missing=True to proceed")

        def take(genes_list):
            """Query counts in the order of genes_list, zero-filling absent genes (CSR, float32)."""
            rows, cols = [], []
            for k, g in enumerate(genes_list):
                j = qpos.get(g)
                if j is not None:
                    rows.append(j); cols.append(k)
            S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(C.shape[1], len(genes_list)))
            return (C @ S).tocsr()

        t = time.time()
        qmask = np.zeros(C.shape[1], dtype=np.float32)
        for g in self.genes:
            j = qpos.get(g)
            if j is not None:
                qmask[j] = 1.0
        totals = np.asarray(C @ qmask).ravel()
        report["zero_total_cells"] = int((totals == 0).sum())
        X3 = take(self.hvg_pca)
        Xn = log_normalize(X3, totals)
        Xd = Xn.toarray()
        emb = {"Python_PCA": self.pca.transform(Xd).astype(np.float32)}
        T["pca_s"] = round(time.time() - t, 2)
        t = time.time()
        emb["NMF"] = self.nmf.transform(Xd if self.nmf_backend == "torch" else Xn).astype(np.float32)
        T["nmf_s"] = round(time.time() - t, 2)
        del Xd, Xn, X3

        # scVI: scArches surgery with the query donors as new batches
        t = time.time()
        X2 = take(self.hvg_scvi)
        report["zero_scvi_count_cells"] = int((np.asarray(X2.sum(1)).ravel() == 0).sum())
        donors = query.obs[donor_key].astype(str).values
        qa = ad.AnnData(X=X2, obs=pd.DataFrame({"scvi_batch": [QUERY_BATCH_PREFIX + d for d in donors]},
                                               index=np.asarray(query.obs_names).astype(str)),
                        var=pd.DataFrame(index=self.hvg_scvi))
        ref = self.scvi_model if self.scvi_model is not None else self.scvi_dir
        scvi.settings.seed = seed
        if self.device.startswith("cuda"):
            gi = int(self.device.split(":")[1])
            tk, lk = dict(accelerator="gpu", devices=[gi]), dict(accelerator="gpu", device=[gi])   # Lightning reads devices=0 as "no device"
        else:
            tk, lk = dict(accelerator="cpu"), dict(accelerator="cpu")
        scvi.model.SCVI.prepare_query_anndata(qa, ref)
        qm = scvi.model.SCVI.load_query_data(qa, ref, **lk)
        qm.train(max_epochs=surgery_epochs, plan_kwargs=dict(SURGERY["plan_kwargs"]), **tk)
        emb["scVI"] = qm.get_latent_representation().astype(np.float32)
        T["scvi_surgery_s"] = round(time.time() - t, 2)
        for v in VIEWS:
            assert np.isfinite(emb[v]).all(), f"{v}: non-finite embedding"

        # stack
        t = time.time()
        res = dict(obs_names=np.asarray(query.obs_names).astype(str), classes=np.array(self.classes),
                   compartments=np.array(self.compartments), genes=report, proba={}, label={}, max_prob={},
                   compartment_proba={}, compartment_label={}, view_proba={})
        M = np.zeros((len(self.classes), len(self.compartments)), dtype=np.float32)
        cidx = {c: i for i, c in enumerate(self.compartments)}
        for i, c in enumerate(self.classes):
            M[i, cidx[self.compartment_of[c]]] = 1.0
        for j, (meta, st) in enumerate(self.stackers.items()):
            vp = base_view_probs(st, emb)
            if j == 0:
                res["view_proba"] = {v: np.mean(vp[v], axis=0).astype(np.float32) for v in VIEWS}
            P = stacked_proba(st, vp).astype(np.float32)
            res["proba"][meta] = P
            res["label"][meta] = np.array(self.classes)[P.argmax(1)]
            res["max_prob"][meta] = P.max(1)
            CP = P @ M
            res["compartment_proba"][meta] = CP
            res["compartment_label"][meta] = np.array(self.compartments)[CP.argmax(1)]
        T["stack_s"] = round(time.time() - t, 2)
        T["total_s"] = round(time.time() - t_all, 2)
        res["timings"] = T
        if return_embeddings:
            res["embeddings"] = emb
        if verbose:
            print(f"project timings {T}")
        return res
