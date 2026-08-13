"""Command-line interface for scboost-ensemble.

Subcommands:
  scboost fit               Train the stacking ensemble + predict on test cells
  scboost build-latents     Build minimal h5ad with one obsm entry per latent
  scboost build-concat      Build minimal h5ad with all pairwise + all-N concat embeddings
  scboost build-from-probs  Build minimal h5ad with per-method probability matrices in obsm
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from . import __version__
from .stacking import StackingEnsembleDonorMetaCached
from .inputs import build_latents_h5ad, build_concat_h5ad, build_probs_h5ad
from .splits import stratified_split
from .eval import build_classification_report


# ---------------------------------------------------------------------------
# fit subcommand
# ---------------------------------------------------------------------------

def cmd_fit(args):
    t_start = time.time()
    if args.gpu is not None:
        import torch
        torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    adata = sc.read_h5ad(args.input)
    t_loaded = time.time()

    # Split train/test
    col = None
    val = None
    if args.split_type != "stratified" and args.split_type in adata.obs.columns:
        col = args.split_type
        vals = adata.obs[col].astype(str).unique()
        if args.holdout_value and args.holdout_value in vals:
            val = args.holdout_value
        else:
            counts = adata.obs[col].astype(str).value_counts()
            target = args.test_size * adata.n_obs
            val = counts.iloc[(counts - target).abs().argsort()].index[0]
        adata.obs["split"] = np.where(adata.obs[col].astype(str) == val, "test", "train")
        print(f"Holdout: {col}='{val}' → "
              f"{(adata.obs['split']=='train').sum()} train, "
              f"{(adata.obs['split']=='test').sum()} test")
    else:
        stratified_split(adata, label_key=args.label_key, batch_key=args.batch_key,
                         test_size=args.test_size, random_state=args.seed)

    train = adata[adata.obs["split"] == "train"]
    test  = adata[adata.obs["split"] == "test"]

    # Drop test cells with unseen cell types
    train_types = set(train.obs[args.label_key].unique())
    test_types  = set(test.obs[args.label_key].unique())
    unseen = test_types - train_types
    if unseen:
        keep = test.obs[args.label_key].isin(train_types)
        n_rm = int((~keep).sum())
        print(f"Removing {n_rm} test cells with unseen cell types: {sorted(unseen)}")
        test = test[keep]
    t_split = time.time()

    # Optional meta-learner hyperparameter overrides (None -> use algo defaults)
    meta_params = {}
    if getattr(args, "meta_l1_ratio", None) is not None:
        meta_params["l1_ratio"] = args.meta_l1_ratio
    if getattr(args, "meta_C", None) is not None:
        meta_params["C"] = args.meta_C
    if getattr(args, "meta_alpha", None) is not None:
        meta_params["alpha"] = args.meta_alpha
    meta_params = meta_params or None

    # Optional NN base-learner hyperparameter overrides (None -> algo defaults)
    algo_params = {}
    if getattr(args, "base_hidden", None) is not None:
        algo_params["hidden"] = [int(h) for h in args.base_hidden]
    if getattr(args, "base_dropout", None) is not None:
        algo_params["dropout"] = args.base_dropout
    if getattr(args, "base_weight_decay", None) is not None:
        algo_params["weight_decay"] = args.base_weight_decay
    algo_params = algo_params or None

    stacker = StackingEnsembleDonorMetaCached(
        obsm_keys=args.methods,
        label_key=args.label_key, batch_key=args.batch_key,
        cell_id_key=args.cell_id_key,
        use_batch_feature=not args.no_batch_feature,
        n_folds=args.n_folds, cv_scheme=args.cv_scheme,
        early_stopping_rounds=args.early_stopping_rounds,
        device=device, n_gpus=args.n_gpus,
        meta_algo=args.meta_algo, meta_params=meta_params,
        base_algo=args.base_algo, algo_params=algo_params,
        seed=args.seed,
        base_cache_dir=args.base_cache_dir,
    )
    stacker.fit(train)
    t_fit = time.time()

    stacker.predict(test, key_added="stacking_pred")
    y_true = np.asarray(test.obs[args.label_key])
    y_pred = np.asarray(test.obs["stacking_pred"])
    proba  = stacker.predict_proba(test)
    report_df = build_classification_report(y_true, y_pred, proba,
                                              stacker.label_encoder_, train)
    t_predict = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if col is not None:
        split_tag = f"holdout_{col}_{val}"
    else:
        split_tag = f"stratified_{args.test_size}"
    tag = f"{args.sample_name}_{split_tag}_base-{args.base_algo}_meta-{args.meta_algo}"

    report_df.to_csv(out_dir / f"{tag}_classification_report.tsv",
                     sep="\t", index_label="cell_type")
    pd.DataFrame(stacker.get_method_importance().items(),
                 columns=["method", "importance"]).to_csv(
        out_dir / f"{tag}_method_importance.tsv", sep="\t", index=False)
    stacker.get_training_summary(
        out_path=str(out_dir / f"{tag}_training_summary.csv"))
    adata.obs[["split"]].to_csv(out_dir / f"{tag}_split.tsv",
                                 sep="\t", index_label="cell_barcode")
    if args.save_model:
        joblib.dump(stacker, out_dir / f"{tag}_ensemble.pkl")
        print(f"Model saved → {tag}_ensemble.pkl")

    t_end = time.time()
    pd.DataFrame([
        {"phase": "data_loading",     "time_s": round(t_loaded  - t_start,  1)},
        {"phase": "train_test_split", "time_s": round(t_split   - t_loaded, 1)},
        {"phase": "fit",              "time_s": round(t_fit     - t_split,  1)},
        {"phase": "predict_eval",     "time_s": round(t_predict - t_fit,    1)},
        {"phase": "total",            "time_s": round(t_end     - t_start,  1)},
    ]).to_csv(out_dir / f"{tag}_runtime.tsv", sep="\t", index=False)
    print(f"\ncache: {stacker.cache_stats_}")
    print(f"Saved to {out_dir}/")


# ---------------------------------------------------------------------------
# build subcommands
# ---------------------------------------------------------------------------

def cmd_build_latents(args):
    build_latents_h5ad(args.npz_path, args.reference_h5ad, args.out,
                       split_h5ad=args.split_h5ad,
                       keys=args.keys)


def cmd_build_concat(args):
    build_concat_h5ad(args.npz_path, args.reference_h5ad, args.out,
                      split_h5ad=args.split_h5ad,
                      base_keys=args.base_keys,
                      pair_prefix=args.pair_prefix,
                      all_key=args.all_key,
                      include_pairs=not args.no_pairs,
                      include_all=not args.no_all)


def cmd_build_from_probs(args):
    build_probs_h5ad(args.npz_path, args.reference_h5ad, args.out,
                     split_h5ad=args.split_h5ad,
                     keys=args.keys,
                     prefix=args.prefix)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(prog="scboost",
                                  description="Donor-aware scBoost stacking ensemble for scRNA-seq.")
    p.add_argument("--version", action="version", version=f"scboost-ensemble {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- fit ---
    f = sub.add_parser("fit", help="Train stacking ensemble + predict on test cells")
    f.add_argument("--input", required=True)
    f.add_argument("--sample_name", required=True)
    f.add_argument("--methods", nargs="+", required=True,
                   help="adata.obsm keys to use as input embeddings")
    f.add_argument("--out_dir", required=True)
    f.add_argument("--label_key", default="cell_type")
    f.add_argument("--batch_key", default="batch")
    f.add_argument("--cell_id_key", default=None)
    f.add_argument("--split_type", default="stratified",
                   help="'stratified' OR an adata.obs column for hold-out (e.g. donor_id)")
    f.add_argument("--holdout_value", default=None)
    f.add_argument("--test_size", type=float, default=0.1)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--cv_scheme", default="stratified",
                   choices=["stratified", "donor", "lodo"])
    f.add_argument("--n_folds", type=int, default=5)
    f.add_argument("--early_stopping_rounds", type=int, default=10)
    f.add_argument("--meta_algo", default="xgboost",
                   choices=["xgboost", "logreg_en", "elasticnet_reg",
                            "torch_logreg_en", "torch_elasticnet_reg"],
                   help="Meta-learner. GPU elastic-net (recommended): "
                        "torch_logreg_en (softmax+CE), torch_elasticnet_reg "
                        "(squared loss on one-hot + softmax). xgboost = default "
                        "tree meta. CPU-only: logreg_en (saga), elasticnet_reg.")
    f.add_argument("--meta_l1_ratio", type=float, default=None,
                   help="Elastic-net mix (logreg_en/elasticnet_reg); default 0.5")
    f.add_argument("--meta_C", type=float, default=None,
                   help="Inverse regularization strength (logreg_en); default 1.0")
    f.add_argument("--meta_alpha", type=float, default=None,
                   help="Regularization strength (elasticnet_reg); default 1e-3")
    f.add_argument("--base_algo", default="xgboost",
                   choices=["xgboost", "mlp", "logistic"],
                   help="Base learner: xgboost (default), mlp (2-layer NN), or "
                        "logistic (linear softmax = multinomial logistic regression).")
    f.add_argument("--base_hidden", nargs="+", type=int, default=None,
                   help="MLP hidden layer sizes (default 256 128)")
    f.add_argument("--base_dropout", type=float, default=None,
                   help="MLP dropout (default 0.3)")
    f.add_argument("--base_weight_decay", type=float, default=None,
                   help="NN L2 weight decay (default 1e-4)")
    f.add_argument("--no_batch_feature", action="store_true")
    f.add_argument("--gpu", type=int, default=None)
    f.add_argument("--n_gpus", type=int, default=1)
    f.add_argument("--base_cache_dir", default=None)
    f.add_argument("--save_model", action="store_true")
    f.set_defaults(func=cmd_fit)

    # --- build-latents ---
    bl = sub.add_parser("build-latents", help="Build h5ad with each latent as obsm entry")
    bl.add_argument("--npz_path", required=True)
    bl.add_argument("--reference_h5ad", required=True)
    bl.add_argument("--out", required=True)
    bl.add_argument("--split_h5ad", default=None)
    bl.add_argument("--keys", nargs="+", default=None)
    bl.set_defaults(func=cmd_build_latents)

    # --- build-concat ---
    bc = sub.add_parser("build-concat",
                          help="Build h5ad with pairwise + all-N concat embeddings")
    bc.add_argument("--npz_path", required=True)
    bc.add_argument("--reference_h5ad", required=True)
    bc.add_argument("--out", required=True)
    bc.add_argument("--split_h5ad", default=None)
    bc.add_argument("--base_keys", nargs="+", default=None)
    bc.add_argument("--pair_prefix", default="concat")
    bc.add_argument("--all_key", default="concat_all")
    bc.add_argument("--no_pairs", action="store_true",
                    help="Skip pairwise concats (only build the all-N entry)")
    bc.add_argument("--no_all", action="store_true",
                    help="Skip the all-N entry (only build pairs)")
    bc.set_defaults(func=cmd_build_concat)

    # --- build-from-probs ---
    bp = sub.add_parser("build-from-probs",
                          help="Build h5ad with per-method probability matrices in obsm")
    bp.add_argument("--npz_path", required=True)
    bp.add_argument("--reference_h5ad", required=True)
    bp.add_argument("--out", required=True)
    bp.add_argument("--split_h5ad", default=None)
    bp.add_argument("--keys", nargs="+", default=None)
    bp.add_argument("--prefix", default=None,
                    help="Filter npz keys by prefix (e.g. 'popV_prob_')")
    bp.set_defaults(func=cmd_build_from_probs)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
