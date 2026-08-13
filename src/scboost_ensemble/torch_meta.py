"""GPU elastic-net linear meta-learners (PyTorch) for the 260618 fork.

A single linear model  ``logits = X @ W.T + b``  (W: n_classes x n_features)
trained full-batch on GPU with an elastic-net penalty on W:

    penalty = alpha * ( l1_ratio * ||W||_1 + (1 - l1_ratio) * 0.5 * ||W||_2^2 )

Two data losses, selected by ``loss``:
  * ``"logistic"``   : cross-entropy on softmax(logits)  -> elastic-net
                       multinomial logistic regression   (meta_algo torch_logreg_en)
  * ``"regression"`` : MSE of raw logits vs one-hot target -> literal
                       "linear regression + elastic net"; probabilities are
                       softmax(logits) at predict time   (meta_algo torch_elasticnet_reg)

Why torch: cuML is not in the pipeline env, and XGBoost's gblinear updater is
CPU-only; torch (already a dependency) gives a genuine-GPU, dependency-free,
self-contained linear meta that also preserves the donor-aware early stopping
of the stacking architecture (the in-fold-k validation set is used for ES).

Exposes the minimal classifier surface used downstream: ``fit``,
``predict_proba`` (numpy, columns in ``classes_`` order = 0..n_local-1),
``classes_`` and ``coef_`` (for method-importance).
"""

from __future__ import annotations
import numpy as np


class TorchElasticNetMeta:
    def __init__(self, loss: str = "logistic", l1_ratio: float = 0.5,
                 alpha: float = 1e-4, lr: float = 0.05, max_epochs: int = 1000,
                 patience: int = 20, eval_every: int = 10, tol: float = 1e-5,
                 standardize: bool = True, device: str = "cpu", seed: int = 42):
        assert loss in ("logistic", "regression")
        self.loss = loss
        self.l1_ratio = float(l1_ratio)
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.eval_every = int(eval_every)
        self.tol = float(tol)
        self.standardize = bool(standardize)
        self.device = device
        self.seed = int(seed)

    # ------------------------------------------------------------------
    def _prep(self, X, torch):
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        if self.standardize:
            Xt = (Xt - self._mean_) / self._std_
        return Xt

    def fit(self, X, y, X_va=None, y_va=None):
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

        # feature standardization stats (train only)
        if self.standardize:
            mean = X.mean(0)
            std = X.std(0)
            std[std < 1e-8] = 1.0
            self._mean_ = torch.as_tensor(mean, device=self.device)
            self._std_ = torch.as_tensor(std, device=self.device)

        Xt = self._prep(X, torch)
        yt = torch.as_tensor(y_idx, device=self.device)
        if self.loss == "regression":
            Yoh = F.one_hot(yt, num_classes=C).float()

        has_va = X_va is not None and len(X_va) > 0
        if has_va:
            Xv = self._prep(X_va, torch)
            yv = torch.as_tensor(
                np.array([cls_to_col[v] for v in np.asarray(y_va)], dtype=np.int64),
                device=self.device)

        W = torch.zeros((C, F_dim), device=self.device, requires_grad=True)
        b = torch.zeros(C, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=self.lr)

        def data_loss(logits, target_idx, target_oh=None):
            if self.loss == "logistic":
                return F.cross_entropy(logits, target_idx)
            return ((logits - target_oh) ** 2).mean()

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
            loss = data_loss(logits, yt, Yoh if self.loss == "regression" else None) + penalty()
            loss.backward()
            opt.step()
            self.n_iter_ = epoch + 1

            if has_va and (epoch % self.eval_every == 0 or epoch == self.max_epochs - 1):
                with torch.no_grad():
                    lv = Xv @ W.t() + b
                    if self.loss == "logistic":
                        vloss = F.cross_entropy(lv, yv).item()
                    else:
                        vloss = ((lv - F.one_hot(yv, C).float()) ** 2).mean().item()
                if vloss < best_val - self.tol:
                    best_val = vloss
                    best_state = (W.detach().clone(), b.detach().clone())
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                # no validation: stop on train-loss plateau
                cur = float(loss.item())
                if abs(prev - cur) < self.tol:
                    break
                prev = cur

        if best_state is not None:
            W, b = best_state
        self._W = W.detach()
        self._b = b.detach()
        self.coef_ = self._W.cpu().numpy()        # (n_classes, n_features)
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            Xt = self._prep(X, torch)
            logits = Xt @ self._W.t() + self._b
            proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy().astype(np.float32)
