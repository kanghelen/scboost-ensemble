"""XGBoost GPU / version compat helpers + GlobalProbsWrapper for missing-class folds."""

from __future__ import annotations
import numpy as np
import xgboost as xgb


def xgb_gpu_kw(device: str) -> dict:
    """Version-safe XGBoost GPU keyword args (handles xgb v1 vs v2+ APIs)."""
    use_gpu = device != "cpu"
    xgb_major = int(xgb.__version__.split(".")[0])
    if use_gpu and xgb_major >= 2:
        return dict(tree_method="hist", device=device)
    if use_gpu:
        return dict(tree_method="gpu_hist")
    return dict(tree_method="hist")


def predict_proba_clf(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)


class GlobalProbsWrapper:
    """Wrap a classifier trained on a subset of global classes so that
    ``predict_proba`` returns shape (n, n_global_classes), filling absent
    classes with zero probability. Used when donor-aware CV leaves some rare
    cell types absent from a particular training fold."""

    def __init__(self, clf, present_classes: np.ndarray, n_global_classes: int):
        self.clf = clf
        self.present_classes = np.asarray(present_classes, dtype=int)
        self.n_global_classes = int(n_global_classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        local = self.clf.predict_proba(X)
        out = np.zeros((local.shape[0], self.n_global_classes), dtype=np.float32)
        out[:, self.present_classes] = local
        return out


def default_xgb_params(gpu_kw: dict) -> dict:
    return dict(
        objective="multi:softprob",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        verbosity=0,
        **gpu_kw,
    )
