"""Non-XGBoost meta-learners for the stacking ensemble (260618 fork).

Adds linear meta-learners as a regularized alternative to the XGBoost meta,
to probe meta-level overfitting:

  * ``logreg_en``      : multinomial LogisticRegression with elastic-net penalty
                         (saga solver), preceded by feature standardization.
                         Built via :func:`make_logreg_en` -> sklearn Pipeline.
  * ``elasticnet_reg`` : literal ElasticNet linear regression on one-hot targets
                         (MultiTaskElasticNet), with softmax to map raw scores to
                         class probabilities. Implemented by
                         :class:`ElasticNetProbaClassifier`.

Both expose ``predict_proba`` with columns ordered as ``classes_`` (= 0..n_local-1
because the caller remaps labels to a contiguous local range), so they drop into
``GlobalProbsWrapper`` exactly like the XGBoost meta.
"""

from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, MultiTaskElasticNet


def make_logreg_en(meta_params: dict) -> Pipeline:
    """StandardScaler -> elastic-net multinomial LogisticRegression.

    saga is the only solver supporting penalty='elasticnet'; it is sensitive to
    feature scaling, hence the StandardScaler. Multinomial is the default for
    multiclass targets in sklearn >= 1.5 (no ``multi_class`` arg passed)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=meta_params.get("l1_ratio", 0.5),
            C=meta_params.get("C", 1.0),
            max_iter=meta_params.get("max_iter", 2000),
            tol=meta_params.get("tol", 1e-3),
            n_jobs=meta_params.get("n_jobs", -1),
            verbose=0,
        )),
    ])


class ElasticNetProbaClassifier:
    """Literal elastic-net *linear regression* turned into a probabilistic
    classifier: fit MultiTaskElasticNet on one-hot class targets, then map the
    raw per-class scores to probabilities with a softmax.

    Exposes the minimal sklearn-classifier surface used downstream:
    ``fit``, ``predict_proba``, ``classes_``, and ``coef_``."""

    def __init__(self, l1_ratio: float = 0.5, alpha: float = 1e-3,
                 max_iter: int = 2000, tol: float = 1e-3, standardize: bool = True):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        # one-hot encode targets (columns follow self.classes_ order)
        class_to_col = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), n_classes), dtype=np.float64)
        Y[np.arange(len(y)), [class_to_col[v] for v in y]] = 1.0

        self.scaler_ = StandardScaler().fit(X) if self.standardize else None
        Xs = self.scaler_.transform(X) if self.standardize else X

        self.model_ = MultiTaskElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio,
            max_iter=self.max_iter, tol=self.tol,
        ).fit(Xs, Y)
        self.coef_ = self.model_.coef_          # (n_classes, n_features)
        self.n_iter_ = int(getattr(self.model_, "n_iter_", -1))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X) if self.standardize else X
        raw = self.model_.predict(Xs)           # (n, n_classes), continuous
        raw = np.atleast_2d(raw)
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
