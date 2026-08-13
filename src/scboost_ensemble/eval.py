"""
Shared evaluation and classification report builder for scRNA-seq pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def build_classification_report(y_true, y_pred, proba, label_encoder, train,
                                label_key="cell_type", k=5):
    """Build a classification report DataFrame with top-1 and top-5 accuracy.

    Parameters
    ----------
    y_true : array-like
        True cell-type labels for the test set.
    y_pred : array-like
        Predicted cell-type labels.
    proba : ndarray, shape (n_cells, n_classes)
        Predicted probability matrix from the classifier.
    label_encoder : LabelEncoder
        Fitted label encoder (must have .transform method).
    train : AnnData
        Training AnnData (used only for train_support counts).
    label_key : str
        Column in train.obs for cell-type labels.
    k : int
        Top-k for top-k accuracy (capped at number of classes).

    Returns
    -------
    report_df : DataFrame
        Per-cell-type report with summary rows (accuracy, macro avg, weighted avg).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_enc = label_encoder.transform(y_true)
    k = min(k, proba.shape[1])
    top_k_indices = np.argsort(proba, axis=1)[:, -k:]
    top5_hit = np.any(top_k_indices == y_true_enc[:, None], axis=1)
    top5_acc = top5_hit.mean()
    top1_acc = accuracy_score(y_true, y_pred)

    report_labels = sorted(set(y_true))
    per_ct_top1 = {ct: np.mean(y_pred[y_true == ct] == ct) for ct in report_labels}
    per_ct_top5 = {ct: top5_hit[y_true == ct].mean() for ct in report_labels}
    macro_top1 = np.mean(list(per_ct_top1.values()))
    macro_top5 = np.mean(list(per_ct_top5.values()))

    print(f"Top-1 Accuracy:        {top1_acc:.4f}")
    print(f"Top-5 Accuracy:        {top5_acc:.4f}  (k={k})")
    print(f"Top-5 Accuracy (macro):{macro_top5:.4f}")
    print(f"Macro F1:              {f1_score(y_true, y_pred, average='macro', labels=report_labels):.4f}")
    print(f"Weighted F1:           {f1_score(y_true, y_pred, average='weighted', labels=report_labels):.4f}")
    print(f"({len(report_labels)} cell types with support > 0)")

    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, labels=report_labels,
                              digits=3, zero_division=0, output_dict=True)
    ).T

    if "accuracy" not in report_df.index:
        report_df.loc["accuracy"] = {
            "precision": top1_acc, "recall": top1_acc,
            "f1-score": top1_acc, "support": len(y_true),
        }

    report_df.insert(0, "top5_accuracy", report_df.index.map(lambda x: per_ct_top5.get(x, "")))
    report_df.insert(0, "top1_accuracy", report_df.index.map(lambda x: per_ct_top1.get(x, "")))

    for idx in ["accuracy", "micro avg"]:
        if idx in report_df.index:
            report_df.at[idx, "top1_accuracy"] = top1_acc
            report_df.at[idx, "top5_accuracy"] = top5_acc
    if "macro avg" in report_df.index:
        report_df.at["macro avg", "top1_accuracy"] = macro_top1
        report_df.at["macro avg", "top5_accuracy"] = macro_top5
    if "weighted avg" in report_df.index:
        report_df.at["weighted avg", "top1_accuracy"] = top1_acc
        report_df.at["weighted avg", "top5_accuracy"] = top5_acc

    train_counts = train.obs[label_key].value_counts().to_dict()
    report_df.insert(0, "train_support", report_df.index.map(lambda x: train_counts.get(x, 0)))

    print("\nPer-cell-type classification report:")
    print(report_df.to_string())

    return report_df
