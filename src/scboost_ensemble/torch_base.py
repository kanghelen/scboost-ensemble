"""GPU neural-net base learners (PyTorch) for the 260619 fork.

Replaces the XGBoost base learner (embedding x in R^D -> softmax over C classes):
  * arch="mlp"      : x_tilde -> [Linear->ReLU->Dropout]*len(hidden) -> Linear -> softmax
  * arch="logistic" : x_tilde -> Linear -> softmax  (multinomial logistic regression)

Inputs are standardized x_tilde=(x-mu)/sigma (stats from the training fold).
Trained with minibatch Adam + L2 weight decay and early stopping on the supplied
validation set (the OOF fold, matching the XGBoost base protocol). Picklable via
joblib (stores cpu state_dict + scaler + classes_), so it drops into the existing
base-learner cache and GlobalProbsWrapper unchanged.
"""
from __future__ import annotations
import numpy as np


def _build_module(D, C, arch, hidden, dropout):
    import torch.nn as nn
    if arch == "logistic":
        return nn.Linear(D, C)
    layers, d = [], D
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, C)]
    return nn.Sequential(*layers)


class TorchBaseClassifier:
    def __init__(self, arch="mlp", hidden=(256, 128), dropout=0.3,
                 weight_decay=1e-4, lr=1e-3, batch_size=4096, max_epochs=200,
                 patience=15, tol=1e-4, device="cpu", seed=42):
        assert arch in ("mlp", "logistic")
        self.arch = arch
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.tol = float(tol)
        self.device = device
        self.seed = int(seed)
        self._module = None

    # --- pickling: drop the live module; keep cpu state_dict ---
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_module"] = None
        return s

    def _dev(self):
        import torch
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            return self.device
        return "cpu"

    def _ensure_module(self):
        import torch
        if self._module is None:
            self._module = _build_module(self.D_, len(self.classes_),
                                         self.arch, self.hidden, self.dropout)
            self._module.load_state_dict({k: torch.as_tensor(v)
                                          for k, v in self.state_.items()})
        return self._module

    def fit(self, X, y, X_val=None, y_val=None):
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
        lossf = nn.CrossEntropyLoss()
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

    def predict_proba(self, X):
        import torch
        dev = self._dev()
        model = self._ensure_module().to(dev).eval()
        Xt = torch.as_tensor((np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_, device=dev)
        with torch.no_grad():
            p = torch.softmax(model(Xt), dim=1)
        return p.cpu().numpy().astype(np.float32)
