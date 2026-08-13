# scboost-ensemble

Donor-aware **stacking ensemble** for scRNA-seq cell-type classification, with
two architectural features beyond textbook stacking:

1. **5 leakage-resistant meta-learners** — each meta-learner trains on out-of-fold
   predictions for cells *not* in its donor-group fold, and uses cells *in* its
   fold as the early-stopping validation set. Inference averages predictions
   across all 5 (base-fold → meta) branches. This avoids the standard pitfall
   of meta-learners picking hyperparameters that overfit same-donor cells.
2. **Per-(method, fold) base classifier caching** — fold trainings are keyed by
   `(method, fold, hash(params, label_classes, train_cell_ids))` and stored on
   disk. Cache hits skip training entirely; ~5–10× speedup for combinatorial
   ensemble experiments where the same base learner appears across many runs.

## Install

```bash
# Editable install while iterating
git clone https://github.com/kanghelen/scboost-ensemble
cd scboost-ensemble
pip install -e ".[dev]"

# Optional: UMAP for OOF diagnostic plots; torch for the MLP base learner
# and the elastic-net meta-learners
pip install -e ".[umap,torch,dev]"
```

Requires Python ≥ 3.10. CUDA is optional but recommended (xgboost will use it
automatically when `device="cuda:0"`).

`torch` is imported lazily, so the package installs and runs without it. It is
required only for `--base_algo mlp` and the `torch_*` meta-learners below.

## Choosing base and meta learners

Both levels of the stack are pluggable, and the choice matters more than it
might look — on high-dimensional embeddings (foundation-model latents of 512–2048
dimensions) an MLP base learner can beat XGBoost by 2 percentage points or more,
while on 100-d batch-corrected embeddings the two are close and the winner is
dataset-dependent. Sweep both rather than assuming.

| flag | choices | notes |
|---|---|---|
| `--base_algo` | `xgboost` (default), `mlp`, `logistic` | `mlp` is a 2-layer NN (`--base_hidden`, `--base_dropout`); needs torch |
| `--meta_algo` | `xgboost` (default), `torch_logreg_en`, `torch_elasticnet_reg`, `logreg_en`, `elasticnet_reg` | `torch_*` run on GPU and need torch; `logreg_en` / `elasticnet_reg` are CPU-only scikit-learn |

```bash
scboost fit --input data.h5ad --methods scVI Harmony Scanorama \
  --base_algo mlp --meta_algo xgboost \
  --split_type donor_id --cv_scheme donor --out_dir out/
```

**Reproducibility note.** The `torch_*` meta-learners call `torch.manual_seed()` but
do not set `torch.use_deterministic_algorithms()`, so their results are reproducible
within one environment but can drift by ~0.2 pp across environments (CUDA reduction
order). The XGBoost meta-learner is bit-reproducible given a fixed base-learner cache.
Prefer it when you need exact reproducibility across machines.

## Quickstart (Python API)

```python
import scanpy as sc
from scboost_ensemble import StackingEnsembleDonorMetaCached

adata = sc.read_h5ad("my_data.h5ad")
train = adata[adata.obs["split"] == "train"].copy()
test  = adata[adata.obs["split"] == "test"].copy()

stacker = StackingEnsembleDonorMetaCached(
    obsm_keys=["scVI", "Harmony", "Scanorama"],
    label_key="cell_type",
    batch_key="donor_id",
    cv_scheme="donor",          # donor-grouped folds for both base + meta
    n_folds=5,
    device="cuda:0", n_gpus=1,
    base_cache_dir="./cache",   # populates on miss, reads on hit
)
stacker.fit(train)
preds = stacker.predict(test)
proba = stacker.predict_proba(test)

print(stacker.get_method_importance())
```

## Quickstart (CLI)

```bash
scboost fit \
    --input my_data.h5ad \
    --sample_name run01 \
    --methods scVI Harmony Scanorama \
    --out_dir results/ \
    --split_type donor_id --cv_scheme donor --test_size 0.1 \
    --gpu 0 --n_gpus 1 \
    --base_cache_dir ./cache
```

Outputs:
- `<sample>_classification_report.tsv` — per-cell-type metrics
- `<sample>_method_importance.tsv`
- `<sample>_training_summary.csv` — best iteration per fold
- `<sample>_runtime.tsv` — per-phase wall time
- `<sample>_split.tsv` — train/test assignment

## Helpers for input construction

When inputs come from a third-party annotator (e.g. popV) saved as `.npz`:

```bash
# Each latent → one obsm entry
scboost build-latents \
    --npz_path popv_latents.npz \
    --reference_h5ad source.h5ad \
    --out outputs/scboost_input.h5ad

# Pairwise + all-N concat embeddings (for ablation studies)
scboost build-concat \
    --npz_path popv_latents.npz \
    --reference_h5ad source.h5ad \
    --out outputs/concat_input.h5ad

# Probability matrices → obsm (for direct meta-learner on classifier outputs)
scboost build-from-probs \
    --npz_path popv_probs.npz \
    --reference_h5ad source.h5ad \
    --out outputs/probs_input.h5ad \
    --prefix popV_prob_
```

## Plots

```python
from scboost_ensemble import plots
plots.plot_method_importance(stacker)
plots.plot_training_summary(stacker)
plots.plot_classification_report("results/run01_classification_report.tsv")
plots.plot_oof_meta_features(stacker, train, projection="umap")
```

See `examples/04_plotting.ipynb` for full walkthroughs.

## Examples

| Notebook | Topic |
|---|---|
| `examples/01_basic_stacking.ipynb` | Minimal 2-embedding stacking on a small dataset |
| `examples/02_concat_embeddings.ipynb` | Using `build_concat_h5ad` for ablation studies |
| `examples/03_prob_integration.ipynb` | Direct meta-learner on probability vectors from any annotator |
| `examples/04_plotting.ipynb` | Diagnostic plots incl. OOF meta-feature projection |

## License

MIT
