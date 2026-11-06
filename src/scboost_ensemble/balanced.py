"""Class-balanced loss for the scBoost stacker (Helen, 2026-10-27: the decision should rest on the embedding, not on
how common a class is in the reference). PLAN.md Step 2b.

Every learner of the stacker minimises an unweighted cross-entropy in the vendored engine, so its posterior carries
the training class frequencies. Here each training cell of class c gets the weight

    w_c = (n_c / n_mean) ** (-beta),   rescaled so the mean weight per training cell is 1,

with n_c the cells of class c among the rows that learner trains on and n_mean their mean over the classes present.
beta = 0 is the engine unchanged; beta = 1 gives every class the same total weight (a flat prior).

The vendored engine (md5-pinned) is not edited. What is copied from it, and why:

  * `WeightedTorchBaseClassifier.fit` = `engine/scboost_ensemble/torch_base.py:68-123` (md5 19e65a5e1da449ba5c3c13d920fbb48f)
    with `nn.CrossEntropyLoss(weight=w)` for the training AND the early-stopping validation loss. The engine builds no
    weighted loss and takes no weight argument.
  * `WeightedTorchElasticNetMeta.fit` = `engine/scboost_ensemble/torch_meta.py:54-141` (md5 feb6e19972c3e012e31f75913d63a315)
    with `F.cross_entropy(..., weight=w)` in the same two places.
  * `BalancedLossMixin._fit_one_meta` = `engine/scboost_ensemble/stacking.py:483-522` (md5 c78a0c1b2bd33ca026d0880081122c1e)
    with `sample_weight` / `sample_weight_eval_set` for the xgboost meta and the weighted torch meta above.

How the base learner is substituted: `stacking.py:281` builds `TorchBaseClassifier(...)` by its module-level name, with a
fixed argument list. During `fit` the mixin points `scboost_ensemble.stacking.TorchBaseClassifier` at
`WeightedTorchBaseClassifier` and sets the active beta on that class; both are restored in `finally`. One fit at a time
per process (all base learners of one fit share one beta, also when they run in the engine's thread pool).

Class indices: the engine renumbers labels to the classes present in a fold (`stacking.py:258-267`, `455-467`), and
the torch learners renumber again to `np.unique(y)`. Weights are therefore always computed inside `fit`, on the labels
that learner actually receives, never on the global encoding.

beta = 0 calls the engine's own code path (no weight at all), so it is bit-identical to the engine.
"""
from __future__ import annotations

import numpy as np

from scboost_ensemble import stacking as _stk
from scboost_ensemble.torch_base import TorchBaseClassifier, _build_module
from scboost_ensemble.torch_meta import TorchElasticNetMeta
from scboost_ensemble._xgb_utils import xgb_gpu_kw


def class_weights(y_idx, beta, n_classes):
    """Per-class weights for integer labels 0..n_classes-1: (n_c / mean n)^-beta over the present classes, scaled so
    that the mean weight per cell is 1. Absent classes get weight 0 (they have no rows)."""
    y_idx = np.asarray(y_idx, dtype=np.int64)
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_classes, dtype=np.float64)
    w[present] = (counts[present] / counts[present].mean()) ** (-float(beta))
    w *= counts.sum() / (counts * w).sum()
    return w


class WeightedTorchBaseClassifier(TorchBaseClassifier):
    """The engine's MLP base learner with a class-weighted cross-entropy (training and early-stopping loss)."""

    _active_beta = None          # set by BalancedLossMixin.fit for the engine's fixed-argument constructor call

    def __init__(self, *args, class_weight_beta=None, **kw):
        super().__init__(*args, **kw)
        self.class_weight_beta = (type(self)._active_beta if class_weight_beta is None else float(class_weight_beta))
        self._force_weighted = False          # test hook: run the copied path even at beta = 0 (weights all 1)

    def fit(self, X, y, X_val=None, y_val=None):
        beta = self.class_weight_beta
        assert beta is not None, "WeightedTorchBaseClassifier built without a class_weight_beta"
        if beta == 0 and not self._force_weighted:
            return super().fit(X, y, X_val, y_val)        # the engine's own code: bit-identical
        # ---- engine/scboost_ensemble/torch_base.py:68-123, with weight=w in both losses ----
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        dev = self._dev()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C, self.D_ = len(self.classes_), X.shape[1]
        c2i = {c: i for i, c in enumerate(self.classes_)}
        yl = np.array([c2i[v] for v in y], dtype=np.int64)
        self.class_weights_ = class_weights(yl, beta, C)                      # added
        wt = torch.as_tensor(self.class_weights_, dtype=torch.float32, device=dev)

        mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
        self.mu_, self.sd_ = mu, sd
        Xt = torch.as_tensor((X - mu) / sd, device=dev)
        yt = torch.as_tensor(yl, device=dev)

        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            Xv = torch.as_tensor((np.asarray(X_val, dtype=np.float32) - mu) / sd, device=dev)
            yv = torch.as_tensor(np.array([c2i[v] for v in np.asarray(y_val)], dtype=np.int64), device=dev)

        model = _build_module(self.D_, C, self.arch, self.hidden, self.dropout).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.CrossEntropyLoss(weight=wt)                                 # changed: weight=wt
        n = len(yt)

        best_val, best_state, bad = np.inf, None, 0
        self.n_iter_ = 0
        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad()
                lossf(model(Xt[idx]), yt[idx]).backward()
                opt.step()
            self.n_iter_ = epoch + 1
            if has_val:
                model.eval()
                with torch.no_grad():
                    vloss = lossf(model(Xv), yv).item()
                if vloss < best_val - self.tol:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.state_ = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        self._module = model
        return self


class WeightedTorchElasticNetMeta(TorchElasticNetMeta):
    """The engine's torch elastic-net logistic meta with a class-weighted cross-entropy (training and ES loss)."""

    def __init__(self, *args, class_weight_beta=0.0, **kw):
        super().__init__(*args, **kw)
        self.class_weight_beta = float(class_weight_beta)
        self._force_weighted = False

    def fit(self, X, y, X_va=None, y_va=None):
        beta = self.class_weight_beta
        if (beta == 0 and not self._force_weighted) or self.loss != "logistic":
            assert beta == 0 or self.loss == "logistic", "class weights are defined for the logistic meta only"
            return super().fit(X, y, X_va=X_va, y_va=y_va)
        # ---- engine/scboost_ensemble/torch_meta.py:54-141, with weight=wt in both cross-entropies ----
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        F_dim = X.shape[1]
        cls_to_col = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_to_col[v] for v in y], dtype=np.int64)
        self.class_weights_ = class_weights(y_idx, beta, C)                   # added
        wt = torch.as_tensor(self.class_weights_, dtype=torch.float32, device=self.device)

        if self.standardize:
            mean = X.mean(0)
            std = X.std(0)
            std[std < 1e-8] = 1.0
            self._mean_ = torch.as_tensor(mean, device=self.device)
            self._std_ = torch.as_tensor(std, device=self.device)

        Xt = self._prep(X, torch)
        yt = torch.as_tensor(y_idx, device=self.device)

        has_va = X_va is not None and len(X_va) > 0
        if has_va:
            Xv = self._prep(X_va, torch)
            yv = torch.as_tensor(np.array([cls_to_col[v] for v in np.asarray(y_va)], dtype=np.int64), device=self.device)

        W = torch.zeros((C, F_dim), device=self.device, requires_grad=True)
        b = torch.zeros(C, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=self.lr)

        def penalty():
            l1 = W.abs().sum()
            l2 = 0.5 * (W ** 2).sum()
            return self.alpha * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2)

        best_val = np.inf
        best_state = None
        bad = 0
        prev = np.inf
        self.n_iter_ = 0
        for epoch in range(self.max_epochs):
            opt.zero_grad()
            logits = Xt @ W.t() + b
            loss = F.cross_entropy(logits, yt, weight=wt) + penalty()            # changed: weight=wt
            loss.backward()
            opt.step()
            self.n_iter_ = epoch + 1

            if has_va and (epoch % self.eval_every == 0 or epoch == self.max_epochs - 1):
                with torch.no_grad():
                    lv = Xv @ W.t() + b
                    vloss = F.cross_entropy(lv, yv, weight=wt).item()            # changed: weight=wt
                if vloss < best_val - self.tol:
                    best_val = vloss
                    best_state = (W.detach().clone(), b.detach().clone())
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                cur = float(loss.item())
                if abs(prev - cur) < self.tol:
                    break
                prev = cur

        if best_state is not None:
            W, b = best_state
        self._W = W.detach()
        self._b = b.detach()
        self.coef_ = self._W.cpu().numpy()
        return self


class BalancedLossMixin:
    """Mix in before the engine's stacker: class-balanced losses in all base learners and meta-learners.

    `class_weight_beta` is added to `algo_params`, so it enters the base-cache directory name (`stacking.py:69`) and a
    cache built under another beta is never hit. The engine reads only the keys it knows from `algo_params`
    (`stacking.py:280-287`), so the extra key does not reach the base learner's constructor."""

    def __init__(self, *args, class_weight_beta=0.0, **kw):
        beta = float(class_weight_beta)
        assert beta >= 0, beta
        kw["algo_params"] = {**(kw.get("algo_params") or {}), "class_weight_beta": beta}
        super().__init__(*args, **kw)
        assert self.base_algo in ("mlp", "logistic"), "class weights are implemented for the torch base learners only"
        assert self.meta_algo in ("xgboost", "torch_logreg_en"), "class weights: xgboost or torch_logreg_en meta only"
        self.class_weight_beta = beta

    def fit(self, adata):
        prev = _stk.TorchBaseClassifier
        assert prev is TorchBaseClassifier, f"engine base class already substituted: {prev}"
        _stk.TorchBaseClassifier = WeightedTorchBaseClassifier
        WeightedTorchBaseClassifier._active_beta = self.class_weight_beta
        try:
            return super().fit(adata)
        finally:
            _stk.TorchBaseClassifier = prev
            WeightedTorchBaseClassifier._active_beta = None

    def _fit_one_meta(self, X_tr, y_tr_local, X_va_es, y_va_local, n_local):
        beta = self.class_weight_beta
        if beta == 0:
            return super()._fit_one_meta(X_tr, y_tr_local, X_va_es, y_va_local, n_local)
        w = class_weights(y_tr_local, beta, n_local)          # local labels 0..n_local-1 (stacking.py:455-467)
        if self.meta_algo == "xgboost":
            # ---- stacking.py:489-501, with sample weights on the training and the early-stopping set ----
            import xgboost as xgb
            gpu_kw = xgb_gpu_kw(self.device)
            use_es = len(X_va_es) > 0
            p = {**self.meta_params, "num_class": n_local, **gpu_kw}
            if use_es:
                p["early_stopping_rounds"] = self.early_stopping_rounds
            clf = xgb.XGBClassifier(**p)
            sw = w[np.asarray(y_tr_local, dtype=np.int64)]
            if use_es:
                sw_va = w[np.asarray(y_va_local, dtype=np.int64)]
                clf.fit(X_tr, y_tr_local, sample_weight=sw, eval_set=[(X_va_es, y_va_local)],
                        sample_weight_eval_set=[sw_va], verbose=False)
                return clf, int(clf.best_iteration), f" (weighted beta={beta})"
            clf.fit(X_tr, y_tr_local, sample_weight=sw, verbose=False)
            return clf, int(clf.n_estimators), f" (no ES, weighted beta={beta})"
        # ---- stacking.py:509-522, with the weighted torch meta ----
        mp = self.meta_params
        clf = WeightedTorchElasticNetMeta(
            loss="logistic", l1_ratio=mp.get("l1_ratio", 0.5), alpha=mp.get("alpha", 1e-4),
            lr=mp.get("lr", 0.05), max_epochs=mp.get("max_epochs", 1000),
            patience=mp.get("patience", 20), eval_every=mp.get("eval_every", 10),
            tol=mp.get("tol", 1e-5), standardize=mp.get("standardize", True),
            device=self.device, seed=self.seed, class_weight_beta=beta)
        X_va = X_va_es if len(X_va_es) > 0 else None
        y_va = y_va_local if len(X_va_es) > 0 else None
        clf.fit(X_tr, y_tr_local, X_va=X_va, y_va=y_va)
        return clf, int(clf.n_iter_), f" ({self.meta_algo}, weighted beta={beta})"
