"""Fold assignment for the reference model (PLAN.md Step 1 rules 3-4).

Classes with at least `min_donors` donors: donor-grouped folds (StratifiedGroupKFold on donor),
so every meta-learner is validated on donors it never trained on. Classes with fewer donors keep
their class; their cells are spread round-robin across folds within the donor, so every branch
sees the class and the meta-learner gets real out-of-fold signal. Those classes are flagged
`within_donor`, because their validation is not donor-disjoint.

The engine computes folds inside `StackingEnsembleDonorMetaCached._make_folds` and takes no fold
argument; `HybridFoldsMixin` overrides that one method without editing the engine.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

WITHIN_DONOR = "within_donor"
DONOR_DISJOINT = "donor-disjoint"


def hybrid_donor_folds(labels, donors, n_folds=5, min_donors=2, seed=42):
    """Return (fold array, per-class table).

    labels, donors: 1-d arrays of equal length (strings or codes).
    The per-class table has columns cells, donors, folds, validation.
    """
    labels = np.asarray(labels).astype(str)
    donors = np.asarray(donors).astype(str)
    n = len(labels)
    assert len(donors) == n
    donors_per_class = pd.Series(donors).groupby(labels).nunique()
    rare = set(donors_per_class.index[donors_per_class < min_donors])
    is_rare = np.isin(labels, list(rare))
    fold = np.full(n, -1, dtype=int)

    # Rule 4: donor-grouped, stratified by class, on the cells of classes with >= min_donors donors.
    idx = np.where(~is_rare)[0]
    if len(idx):
        sg = StratifiedGroupKFold(n_splits=n_folds)  # engine defaults: shuffle=False
        for k, (_, va) in enumerate(sg.split(np.zeros(len(idx)), labels[idx], donors[idx])):
            fold[idx[va]] = k
    # Rule 3: round-robin within donor for the rare classes, in a seeded but reproducible order.
    rng = np.random.default_rng(seed)
    for c in sorted(rare):
        cidx = np.where(labels == c)[0]
        for d in np.unique(donors[cidx]):
            didx = cidx[donors[cidx] == d]
            didx = didx[rng.permutation(len(didx))]
            fold[didx] = np.arange(len(didx)) % n_folds
    assert (fold >= 0).all()

    df = pd.DataFrame(dict(label=labels, donor=donors, fold=fold))
    g = df.groupby("label")
    table = pd.concat([g.size().rename("cells"), g.donor.nunique().rename("donors"), g.fold.nunique().rename("folds")], axis=1)
    table["validation"] = np.where(table.index.isin(list(rare)), WITHIN_DONOR, DONOR_DISJOINT)
    return fold, table


def check_fold_properties(labels, donors, fold, table, n_folds=5):
    """Assertions used by the smoke test and by the fit driver."""
    labels = np.asarray(labels).astype(str)
    donors = np.asarray(donors).astype(str)
    df = pd.DataFrame(dict(label=labels, donor=donors, fold=fold))
    for c, row in table.iterrows():
        sub = df[df.label == c]
        if row.validation == WITHIN_DONOR:
            expect = min(n_folds, len(sub))
            assert sub.fold.nunique() == expect, (c, sub.fold.nunique(), expect)
        else:
            # every donor's cells of this class sit in exactly one fold
            per_donor = sub.groupby("donor").fold.nunique()
            assert (per_donor == 1).all(), (c, per_donor[per_donor > 1].to_dict())
    # and, across the donor-disjoint cells, a donor never spans two folds
    dd = df[df.label.isin(table.index[table.validation == DONOR_DISJOINT])]
    spans = dd.groupby("donor").fold.nunique()
    assert (spans == 1).all(), spans[spans > 1].to_dict()
    return True


class HybridFoldsMixin:
    """Mix into StackingEnsembleDonorMetaCached to replace _make_folds.

    The engine calls self._make_folds(labels, batch) with the encoded training labels and the
    batch column (pass batch_key="donor_id"), and consumes a list of (train_idx, val_idx) pairs
    (stacking.py:340-349). `self.min_donors` and `self.seed` are read if present.
    """

    def _make_folds(self, labels, batch):
        fold, table = hybrid_donor_folds(labels, batch, n_folds=self.n_folds,
                                         min_donors=getattr(self, "min_donors", 2),
                                         seed=getattr(self, "seed", 42))
        check_fold_properties(labels, batch, fold, table, n_folds=self.n_folds)
        self.fold_table_ = table
        print(f"  CV: hybrid donor folds (n={self.n_folds}; "
              f"{int((table.validation == WITHIN_DONOR).sum())} within-donor classes)")
        return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(self.n_folds)]
