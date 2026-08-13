"""
Donor-aware stacking ensemble (noLeak meta architecture) with per-fold base
classifier caching.

Key architectural choices:

1. **5 meta-learners with donor-aware early stopping** — fixes the standard
   stacking-ensemble meta-learner ES-leakage where a random meta validation
   split places same-donor cells on both sides. Each meta-learner k trains on
   OOF predictions for cells NOT in outer fold k (donor-grouped) and uses
   cells IN fold k as the ES validation set. At inference, base learner fold
   k feeds meta-learner k, and the 5 fold predictions are averaged.

2. **Per-(method, algo, fold) base classifier caching** — fold trainings are
   keyed by hashes of (params, label classes, train cell IDs) and stored on
   disk. Cache hits skip training entirely; large speedups for permutation /
   combinatorial experiments where the same base learner appears in many
   ensemble configurations.

Public class: ``StackingEnsembleDonorMetaCached``.
"""

from __future__ import annotations

import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import joblib
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from anndata import AnnData

from ._xgb_utils import (
    xgb_gpu_kw,
    predict_proba_clf,
    GlobalProbsWrapper,
    default_xgb_params,
)
from .meta_learners import make_logreg_en, ElasticNetProbaClassifier
from .torch_meta import TorchElasticNetMeta
from .torch_base import TorchBaseClassifier

META_ALGOS = ("xgboost", "logreg_en", "elasticnet_reg",
              "torch_logreg_en", "torch_elasticnet_reg")


# ---------------------------------------------------------------------------
# Cache key construction
# ---------------------------------------------------------------------------

def _short_sha(obj_str: str, n: int = 16) -> str:
    return hashlib.sha1(obj_str.encode()).hexdigest()[:n]


def _build_cache_dirname(
    key: str, algo: str, fold_k: int,
    cv_scheme: str, n_folds: int, early_stop: int,
    algo_params: dict, label_classes: list, train_cell_ids: list,
) -> str:
    params_hash  = _short_sha(json.dumps(algo_params, sort_keys=True, default=str))
    classes_hash = _short_sha(",".join(map(str, sorted(label_classes))))
    train_hash   = _short_sha(",".join(map(str, sorted(map(str, train_cell_ids)))))
    safe_key = key.replace(":", "_").replace("+", "-")
    return (f"{safe_key}__{algo}__fold{fold_k}__{cv_scheme}{n_folds}__"
            f"es{early_stop}__p{params_hash}__c{classes_hash}__t{train_hash}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StackingEnsembleDonorMetaCached:
    """Donor-aware stacking ensemble with 5 meta-learners + base classifier cache."""

    def __init__(
        self,
        obsm_keys: List[str],
        label_key: str = "cell_type",
        batch_key: str = "batch",
        cell_id_key: Optional[str] = None,
        use_batch_feature: bool = True,
        n_folds: int = 5,
        cv_scheme: str = "stratified",
        early_stopping_rounds: int = 10,
        device: str = "cpu",
        n_gpus: int = 1,
        algo_params: Optional[dict] = None,
        meta_params: Optional[dict] = None,
        meta_algo: str = "xgboost",
        base_algo: str = "xgboost",
        seed: int = 42,
        base_cache_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        obsm_keys : list of str
            adata.obsm keys to use as input embeddings (one per base learner).
        label_key, batch_key : str
            adata.obs columns for cell-type labels and batch/donor groups.
        cell_id_key : str or None
            adata.obs column with unique cell IDs used in the cache key.
            If None, falls back to adata.obs.index.
        cv_scheme : {"stratified", "donor", "lodo"}
            Cross-validation strategy. "donor" uses StratifiedGroupKFold by
            ``batch_key`` (recommended for donor-holdout evaluation).
        base_cache_dir : str or None
            Directory for caching base classifier outputs across runs.
            If None, caching is disabled.
        """
        self.obsm_keys             = list(obsm_keys)
        self.label_key             = label_key
        self.batch_key             = batch_key
        self.cell_id_key           = cell_id_key
        self.use_batch_feature     = use_batch_feature
        self.n_folds               = n_folds
        self.cv_scheme             = cv_scheme
        self.early_stopping_rounds = early_stopping_rounds
        self.n_gpus                = n_gpus
        self.seed                  = seed
        self.base_cache_dir        = Path(base_cache_dir) if base_cache_dir else None

        if device.startswith("cuda"):
            parts = device.split(":")
            self.gpu_start = int(parts[1]) if len(parts) > 1 else 0
        else:
            self.gpu_start = 0
        self.device = device

        if base_algo not in ("xgboost", "mlp", "logistic"):
            raise ValueError(f"base_algo must be xgboost|mlp|logistic, got {base_algo!r}")
        self.base_algo = base_algo
        gpu_kw = xgb_gpu_kw(device)
        self.algo = base_algo
        if base_algo == "xgboost":
            self.algo_params = {**default_xgb_params(gpu_kw), **(algo_params or {})}
        else:
            # device-free params -> cache key is GPU-independent (any GPU hits)
            nn_default = dict(arch=base_algo, dropout=0.3, weight_decay=1e-4,
                              lr=1e-3, batch_size=4096, max_epochs=200,
                              patience=15, tol=1e-4)
            if base_algo == "mlp":
                nn_default["hidden"] = [256, 128]
            self.algo_params = {**nn_default, **(algo_params or {})}

        if meta_algo not in META_ALGOS:
            raise ValueError(f"meta_algo must be one of {META_ALGOS}, got {meta_algo!r}")
        self.meta_algo = meta_algo
        if meta_algo == "xgboost":
            default_meta = dict(
                objective="multi:softprob", max_depth=4, learning_rate=0.05,
                n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="mlogloss", verbosity=0, **gpu_kw,
            )
        elif meta_algo == "logreg_en":
            default_meta = dict(l1_ratio=0.5, C=1.0, max_iter=2000, tol=1e-3, n_jobs=-1)
        elif meta_algo == "elasticnet_reg":
            default_meta = dict(l1_ratio=0.5, alpha=1e-3, max_iter=2000, tol=1e-3,
                                standardize=True)
        elif meta_algo in ("torch_logreg_en", "torch_elasticnet_reg"):
            default_meta = dict(l1_ratio=0.5, alpha=1e-4, lr=0.05, max_epochs=1000,
                                patience=20, eval_every=10, tol=1e-5, standardize=True)
        else:
            default_meta = {}
        # user-supplied meta_params override individual defaults
        self.meta_params = {**default_meta, **(meta_params or {})}

        # State (filled by .fit())
        self.n_cell_types: Optional[int]               = None
        self.label_encoder_                            = LabelEncoder()
        self.base_classifiers_: Dict[str, list]        = {}
        self.base_best_iterations_: Dict[str, list]    = {}
        self.fold_assignments_: Optional[np.ndarray]   = None
        self.meta_classifiers_: List                   = []
        self.meta_best_iterations_: List[int]          = []
        self._batch_enc: Optional[OneHotEncoder]       = None
        self.train_support_: Optional[Dict[str, int]]  = None
        self.cache_stats_                              = {"hits": 0, "misses": 0}

        print(f"[StackingDonorMetaCached] obsm_keys={self.obsm_keys}, "
              f"cv_scheme={cv_scheme}, n_folds={n_folds}, "
              f"device={device}, n_gpus={n_gpus}, "
              f"cache_dir={self.base_cache_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract(self, adata: AnnData):
        embs = {k: np.asarray(adata.obsm[k]) for k in self.obsm_keys}
        labels = np.asarray(adata.obs[self.label_key])
        batch = np.asarray(adata.obs[self.batch_key])
        if self.cell_id_key and self.cell_id_key in adata.obs.columns:
            cell_ids = np.asarray(adata.obs[self.cell_id_key]).astype(str)
        else:
            cell_ids = np.asarray(adata.obs.index).astype(str)
        return embs, labels, batch, cell_ids

    def _make_xgb(self, gpu_id: int, num_class: int, early_stopping: bool):
        device_str = f"cuda:{gpu_id}" if self.device != "cpu" else "cpu"
        gpu_kw = xgb_gpu_kw(device_str)
        kw = {**self.algo_params, "num_class": num_class, **gpu_kw}
        if early_stopping:
            return xgb.XGBClassifier(**kw,
                                      early_stopping_rounds=self.early_stopping_rounds)
        return xgb.XGBClassifier(**kw)

    # ------------------------------------------------------------------
    # Per-fold base learner training (with cache lookup)
    # ------------------------------------------------------------------

    def _train_one_base_fold(
        self,
        key: str, fold_k: int, X: np.ndarray, labels: np.ndarray,
        tr_idx: np.ndarray, va_idx: np.ndarray, gpu_id: int,
        train_cell_ids: list, label_classes: list,
    ):
        cache_status = "disabled"
        cache_dir = None
        if self.base_cache_dir is not None:
            cache_name = _build_cache_dirname(
                key=key, algo=self.algo, fold_k=fold_k,
                cv_scheme=self.cv_scheme, n_folds=self.n_folds,
                early_stop=self.early_stopping_rounds,
                algo_params=self.algo_params, label_classes=label_classes,
                train_cell_ids=train_cell_ids,
            )
            cache_dir = self.base_cache_dir / cache_name
            model_path = cache_dir / "model.joblib"
            va_path    = cache_dir / "va_proba.npz"
            if model_path.exists() and va_path.exists():
                try:
                    clf = joblib.load(model_path)
                    npz = np.load(va_path, allow_pickle=False)
                    va_proba = npz["proba"]
                    best_iter = int(npz["best_iter"])
                    return clf, va_proba, best_iter, "hit"
                except Exception as e:
                    print(f"  [cache] WARN: corrupt cache at {cache_dir}; rebuilding ({e})")
            cache_status = "miss"

        # ---- Train ----
        n_classes = len(label_classes)
        present = np.unique(labels[tr_idx])
        n_local = len(present)
        needs_remap = n_local < n_classes

        if needs_remap:
            g2l = {int(g): l for l, g in enumerate(present)}
            tr_labels = np.array([g2l[int(c)] for c in labels[tr_idx]])
            va_mask   = np.isin(labels[va_idx], present)
            va_es     = va_idx[va_mask]
            va_es_labels = np.array([g2l[int(c)] for c in labels[va_es]])
        else:
            tr_labels = labels[tr_idx]
            va_es     = va_idx
            va_es_labels = labels[va_idx]

        use_es = len(va_es) > 0
        if self.base_algo == "xgboost":
            clf = self._make_xgb(gpu_id, num_class=n_local, early_stopping=use_es)
            if use_es:
                clf.fit(X[tr_idx], tr_labels,
                        eval_set=[(X[va_es], va_es_labels)], verbose=False)
                best_iter = int(clf.best_iteration)
            else:
                clf.fit(X[tr_idx], tr_labels, verbose=False)
                best_iter = int(clf.n_estimators)
        else:  # mlp / logistic torch base learner (ES on the OOF fold, like XGBoost)
            p = self.algo_params
            clf = TorchBaseClassifier(
                arch=p["arch"], hidden=tuple(p.get("hidden", (256, 128))),
                dropout=p["dropout"], weight_decay=p["weight_decay"], lr=p["lr"],
                batch_size=p["batch_size"], max_epochs=p["max_epochs"],
                patience=p["patience"], tol=p["tol"],
                device=f"cuda:{gpu_id}" if self.device.startswith("cuda") else "cpu",
                seed=self.seed)
            clf.fit(X[tr_idx], tr_labels,
                    X[va_es] if use_es else None, va_es_labels if use_es else None)
            best_iter = int(clf.n_iter_)

        if needs_remap:
            wrapped = GlobalProbsWrapper(clf, present, n_classes)
            va_proba = wrapped.predict_proba(X[va_idx]) if len(va_idx) else np.zeros((0, n_classes))
            clf = wrapped
        else:
            va_proba = predict_proba_clf(clf, X[va_idx]) if len(va_idx) else np.zeros((0, n_classes))

        # ---- Cache write ----
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                joblib.dump(clf, cache_dir / "model.joblib", compress=3)
                np.savez_compressed(cache_dir / "va_proba.npz",
                                    proba=va_proba.astype(np.float32),
                                    best_iter=np.int64(best_iter))
                with open(cache_dir / "manifest.json", "w") as f:
                    json.dump({
                        "embedding": key, "algo": self.algo, "fold_k": fold_k,
                        "cv_scheme": self.cv_scheme, "n_folds": self.n_folds,
                        "early_stopping_rounds": self.early_stopping_rounds,
                        "n_classes": n_classes,
                        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
                        "best_iteration": best_iter,
                    }, f, indent=2)
            except Exception as e:
                print(f"  [cache] WARN: could not write cache to {cache_dir}: {e}")

        return clf, va_proba, best_iter, cache_status

    # ------------------------------------------------------------------
    # OOF orchestration
    # ------------------------------------------------------------------

    def _make_folds(self, labels: np.ndarray, batch: np.ndarray):
        if self.cv_scheme == "donor":
            cv = StratifiedGroupKFold(n_splits=self.n_folds)
            folds = list(cv.split(np.zeros((len(labels), 1)), labels, groups=batch))
            print(f"  CV: StratifiedGroupKFold(n={self.n_folds}, groups=batch)")
        elif self.cv_scheme == "lodo":
            cv = LeaveOneGroupOut()
            folds = list(cv.split(np.zeros((len(labels), 1)), labels, groups=batch))
            print(f"  CV: LeaveOneGroupOut ({len(folds)} folds, groups=batch)")
        else:
            cv = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
            folds = list(cv.split(np.zeros((len(labels), 1)), labels))
            print(f"  CV: StratifiedKFold(n={self.n_folds})")
        return folds

    def _oof_with_folds(self, embs, labels, batch, cell_ids):
        n, C = len(labels), self.n_cell_types
        folds = self._make_folds(labels, batch)
        if len(folds) != self.n_folds:
            self.n_folds = len(folds)

        fold_assignments = np.full(n, -1, dtype=np.int64)
        for k, (_, va) in enumerate(folds):
            fold_assignments[va] = k
        if (fold_assignments < 0).any():
            raise RuntimeError("Some cells not assigned to any fold (CV bug?)")

        meta_oof = np.zeros((n, len(self.obsm_keys) * C), dtype=np.float32)
        train_cell_ids = list(map(str, cell_ids))
        label_classes  = list(map(str, self.label_encoder_.classes_))

        work = [(ki, key, fk) for ki, key in enumerate(self.obsm_keys)
                for fk in range(self.n_folds)]
        print(f"  Base learner trainings: {len(work)} "
              f"= {len(self.obsm_keys)} embeddings × {self.n_folds} folds")

        for key in self.obsm_keys:
            self.base_classifiers_[key] = [None] * self.n_folds
            self.base_best_iterations_[key] = [None] * self.n_folds

        use_parallel = self.device != "cpu" and self.n_gpus > 1
        if use_parallel:
            t_par = time.time()
            with ThreadPoolExecutor(max_workers=self.n_gpus) as pool:
                futures = {}
                for wi, (ki, key, fk) in enumerate(work):
                    tr, va = folds[fk]
                    gpu_id = self.gpu_start + (wi % self.n_gpus)
                    fut = pool.submit(self._train_one_base_fold,
                                      key, fk, embs[key], labels, tr, va, gpu_id,
                                      train_cell_ids, label_classes)
                    futures[fut] = (ki, key, fk, va)
                for fut in as_completed(futures):
                    ki, key, fk, va = futures[fut]
                    clf, va_proba, best_iter, cstat = fut.result()
                    self.base_classifiers_[key][fk] = clf
                    self.base_best_iterations_[key][fk] = best_iter
                    cs, ce = ki * C, (ki + 1) * C
                    meta_oof[va, cs:ce] = va_proba.astype(np.float32)
                    self.cache_stats_["hits" if cstat == "hit" else "misses"] += (cstat in ("hit", "miss"))
                    print(f"    [{cstat:5s}] {key} fold{fk} best_iter={best_iter}")
            print(f"  All base classifiers done ({time.time()-t_par:.1f}s wall)")
        else:
            for wi, (ki, key, fk) in enumerate(work):
                tr, va = folds[fk]
                clf, va_proba, best_iter, cstat = self._train_one_base_fold(
                    key, fk, embs[key], labels, tr, va,
                    self.gpu_start, train_cell_ids, label_classes)
                self.base_classifiers_[key][fk] = clf
                self.base_best_iterations_[key][fk] = best_iter
                cs, ce = ki * C, (ki + 1) * C
                meta_oof[va, cs:ce] = va_proba.astype(np.float32)
                self.cache_stats_["hits" if cstat == "hit" else "misses"] += (cstat in ("hit", "miss"))
                print(f"    [{cstat:5s}] {key} fold{fk} best_iter={best_iter}")

        return meta_oof, fold_assignments

    # ------------------------------------------------------------------
    # Test-time meta features (per fold)
    # ------------------------------------------------------------------

    def _test_meta_for_fold(self, embs, fold_k: int) -> np.ndarray:
        C = self.n_cell_types
        n_test = next(iter(embs.values())).shape[0]
        meta = np.zeros((n_test, len(self.obsm_keys) * C), dtype=np.float32)
        for ki, key in enumerate(self.obsm_keys):
            clf = self.base_classifiers_[key][fold_k]
            cs, ce = ki * C, (ki + 1) * C
            meta[:, cs:ce] = predict_proba_clf(clf, embs[key]).astype(np.float32)
        return meta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, adata: AnnData):
        """Train base classifiers (with caching) + 5 donor-aware meta-learners."""
        t_start = time.time()
        embs, labels, batch, cell_ids = self._extract(adata)
        enc_labels = self.label_encoder_.fit_transform(labels)
        self.n_cell_types = len(self.label_encoder_.classes_)

        unique, counts = np.unique(labels, return_counts=True)
        self.train_support_ = dict(zip(unique, counts.astype(int)))
        print(f"Stacking fit: {adata.n_obs} cells, {self.n_cell_types} cell types, "
              f"{len(self.obsm_keys)} embeddings × {self.n_folds} folds "
              f"= {len(self.obsm_keys) * self.n_folds} base learners")

        meta_oof, fold_assignments = self._oof_with_folds(embs, enc_labels, batch, cell_ids)
        self.fold_assignments_ = fold_assignments

        if self.use_batch_feature:
            self._batch_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
                batch.reshape(-1, 1))
            batch_oh = self._batch_enc.transform(batch.reshape(-1, 1)).astype(np.float32)
            meta_oof_full = np.hstack([meta_oof, batch_oh])
        else:
            meta_oof_full = meta_oof

        print(f"\nTraining {self.n_folds} meta-learners (algo={self.meta_algo}, "
              f"{meta_oof_full.shape[1]} features each)...")
        self.meta_classifiers_ = []
        self.meta_best_iterations_ = []
        C = self.n_cell_types
        for k in range(self.n_folds):
            in_fold_k = (fold_assignments == k)
            tr_mask = ~in_fold_k
            X_tr, X_va = meta_oof_full[tr_mask], meta_oof_full[in_fold_k]
            y_tr, y_va = enc_labels[tr_mask], enc_labels[in_fold_k]

            present = np.unique(y_tr)
            n_local = len(present)
            needs_remap = n_local < C
            if needs_remap:
                g2l = {int(g): l for l, g in enumerate(present)}
                y_tr_local = np.array([g2l[int(c)] for c in y_tr])
                va_present_mask = np.isin(y_va, present)
                X_va_es = X_va[va_present_mask]
                y_va_local = np.array([g2l[int(c)] for c in y_va[va_present_mask]])
            else:
                y_tr_local = y_tr
                X_va_es = X_va
                y_va_local = y_va

            meta_clf, best_iter, note = self._fit_one_meta(
                X_tr, y_tr_local, X_va_es, y_va_local, n_local)
            if needs_remap:
                meta_clf = GlobalProbsWrapper(meta_clf, present, C)
            self.meta_classifiers_.append(meta_clf)
            self.meta_best_iterations_.append(best_iter)
            print(f"  Meta {k+1}/{self.n_folds}: train={tr_mask.sum()}, "
                  f"val={in_fold_k.sum()}, n_local={n_local}/{C}, "
                  f"best_iter={best_iter}{note}")

        print(f"\nFit complete ({time.time()-t_start:.1f}s)  "
              f"cache: {self.cache_stats_['hits']} hits, {self.cache_stats_['misses']} misses")
        return self

    def _fit_one_meta(self, X_tr, y_tr_local, X_va_es, y_va_local, n_local):
        """Fit one meta-learner per ``self.meta_algo``.

        Returns (fitted_estimator, best_iter_for_reporting, log_note). Only the
        XGBoost path uses the in-fold-k validation set (donor-aware early
        stopping); the linear metas regularize via their penalty instead."""
        if self.meta_algo == "xgboost":
            gpu_kw = xgb_gpu_kw(self.device)
            use_es = len(X_va_es) > 0
            p = {**self.meta_params, "num_class": n_local, **gpu_kw}
            if use_es:
                p["early_stopping_rounds"] = self.early_stopping_rounds
            clf = xgb.XGBClassifier(**p)
            if use_es:
                clf.fit(X_tr, y_tr_local,
                        eval_set=[(X_va_es, y_va_local)], verbose=False)
                return clf, int(clf.best_iteration), ""
            clf.fit(X_tr, y_tr_local, verbose=False)
            return clf, int(clf.n_estimators), " (no ES)"

        if self.meta_algo == "logreg_en":
            clf = make_logreg_en(self.meta_params)
            clf.fit(X_tr, y_tr_local)
            n_iter = int(np.max(clf.named_steps["clf"].n_iter_))
            return clf, n_iter, " (logreg_en)"

        if self.meta_algo in ("torch_logreg_en", "torch_elasticnet_reg"):
            mp = self.meta_params
            loss = "logistic" if self.meta_algo == "torch_logreg_en" else "regression"
            clf = TorchElasticNetMeta(
                loss=loss, l1_ratio=mp.get("l1_ratio", 0.5), alpha=mp.get("alpha", 1e-4),
                lr=mp.get("lr", 0.05), max_epochs=mp.get("max_epochs", 1000),
                patience=mp.get("patience", 20), eval_every=mp.get("eval_every", 10),
                tol=mp.get("tol", 1e-5), standardize=mp.get("standardize", True),
                device=self.device, seed=self.seed)
            # donor-aware early stopping: use in-fold-k cells as validation
            X_va = X_va_es if len(X_va_es) > 0 else None
            y_va = y_va_local if len(X_va_es) > 0 else None
            clf.fit(X_tr, y_tr_local, X_va=X_va, y_va=y_va)
            return clf, int(clf.n_iter_), f" ({self.meta_algo})"

        # elasticnet_reg
        mp = self.meta_params
        clf = ElasticNetProbaClassifier(
            l1_ratio=mp.get("l1_ratio", 0.5), alpha=mp.get("alpha", 1e-3),
            max_iter=mp.get("max_iter", 2000), tol=mp.get("tol", 1e-3),
            standardize=mp.get("standardize", True))
        clf.fit(X_tr, y_tr_local)
        return clf, int(clf.n_iter_), " (elasticnet_reg)"

    def predict_proba(self, adata: AnnData) -> np.ndarray:
        """Average predictions from the 5 (base_fold_k → meta_k) branches."""
        embs, _, batch, _ = self._extract(adata)
        batch_oh = None
        if self.use_batch_feature and self._batch_enc is not None:
            batch_oh = self._batch_enc.transform(batch.reshape(-1, 1)).astype(np.float32)
        probas = []
        for k in range(self.n_folds):
            X_meta_k = self._test_meta_for_fold(embs, k)
            if batch_oh is not None:
                X_meta_k = np.hstack([X_meta_k, batch_oh])
            probas.append(self.meta_classifiers_[k].predict_proba(X_meta_k))
        return np.mean(probas, axis=0)

    def predict(self, adata: AnnData, key_added: str = "stacking_pred") -> np.ndarray:
        t0 = time.time()
        proba = self.predict_proba(adata)
        preds = self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))
        adata.obs[key_added] = preds
        print(f"Predict complete on {adata.n_obs} cells ({time.time()-t0:.1f}s)")
        return preds

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _meta_feature_importance(inner) -> np.ndarray:
        """Per-feature importance vector for any supported meta-learner.

        XGBoost -> ``feature_importances_``; linear metas (LogisticRegression
        pipeline / ElasticNetProbaClassifier) -> sum of |coef| over classes."""
        if hasattr(inner, "feature_importances_"):           # xgboost
            return np.asarray(inner.feature_importances_)
        coef = None
        if hasattr(inner, "named_steps"):                    # logreg_en pipeline
            coef = inner.named_steps["clf"].coef_
        elif hasattr(inner, "coef_"):                        # elasticnet_reg
            coef = inner.coef_
        if coef is None:
            raise AttributeError("meta-learner exposes no importance/coef_")
        return np.abs(np.atleast_2d(coef)).sum(axis=0)       # (n_features,)

    def get_method_importance(self) -> Dict[str, float]:
        """Mean importance across the 5 meta-learners, summed within each method."""
        C = self.n_cell_types
        scores = {key: 0.0 for key in self.obsm_keys}
        for meta_clf in self.meta_classifiers_:
            inner = meta_clf.clf if isinstance(meta_clf, GlobalProbsWrapper) else meta_clf
            imp = self._meta_feature_importance(inner)
            for ki, key in enumerate(self.obsm_keys):
                scores[key] += float(imp[ki * C:(ki + 1) * C].sum())
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def get_training_summary(self, out_path: Optional[str] = None) -> dict:
        import pandas as pd
        rows = []
        for key in self.obsm_keys:
            for fold_i, best_iter in enumerate(self.base_best_iterations_[key]):
                rows.append({"component": "base", "embedding": key, "algo": self.algo,
                             "fold": fold_i, "best_iteration": best_iter})
        for k, best_iter in enumerate(self.meta_best_iterations_):
            rows.append({"component": "meta", "embedding": "_meta_", "algo": self.meta_algo,
                         "fold": k, "best_iteration": best_iter})

        summary = {
            "base_best_iterations": {k: list(map(int, v))
                                      for k, v in self.base_best_iterations_.items()},
            "meta_best_iterations": list(map(int, self.meta_best_iterations_)),
            "meta_algo": self.meta_algo,
            "cache_stats": self.cache_stats_,
            "n_folds": self.n_folds,
            "n_cell_types": self.n_cell_types,
        }
        if out_path is not None:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"Training summary saved to {out_path}")
        return summary
