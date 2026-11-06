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

## Base-learner cache

Each (embedding, fold) training is stored under a directory named for everything
that determines it:

```
Geoksketch__xgboost__fold0__d100__donor5__es10__p<params>__c<classes>__t<train_ids>
                            ^^^^ input width
```

The `d{N}` width segment (added in 0.3.0) is what keeps a 100-d embedding and a
50-d one from colliding when they share an obsm key name. Two further guards sit
behind it:

- **Width guard.** Before a cached model is used, its expected input width is
  compared against the embedding in hand; a mismatch retrains rather than failing
  at predict time.
- **Content fingerprint.** Each entry records a `content_sha` of the matrix it was
  trained on. If an embedding changes while keeping both its name *and* its width,
  the run warns. Pass `--strict_cache` to treat that as a miss instead —
  recommended for any run whose numbers you publish, since a warning is otherwise
  easy to miss in a long sweep log.

**The naming discipline that avoids all of this:** give each version of an
embedding its own obsm key (`Python_PCA_50d`, `Python_PCA_100d`) rather than
re-deriving one in place under an existing name.

Entries written before 0.3.0 carry no width segment. They are reused when the
model inside matches the current width, and symlinked under the new name
(counted as `legacy_adopted` within `cache_stats_["hits"]`); `--legacy_cache
ignore` retrains instead. Legacy entries have no fingerprint, so a same-width
content change in one of them cannot be detected — use a fresh cache directory
with `--strict_cache` when that matters.

## Pretrained reference model

`scboost_ensemble.reference_model` (added in 0.3.0) fits scBoost once on a labelled
reference, saves every fitted object, and places new query cells without refitting
anything on them. Three embeddings can embed new cells, so the model uses those
three:

- **PCA**: 100 components on log1p(counts normalised to 10,000) of 3,000 reference HVGs.
- **NMF**: 100 components on the same matrix. The gene basis is kept, and query cells are
  solved against it.
- **scVI**: 30 latent dimensions on 2,000 reference HVGs. Query cells are placed by
  scArches surgery, with each query donor added as a new batch.

On these three embeddings sit 3 (embeddings) × 5 (folds) = 15 MLP base classifiers and
5 meta-learners, one per fold. The folds are grouped by donor (`scboost_ensemble.folds`).

The model trained on Tabula Sapiens 2.0 (637,267 cells, 167 cell types) is
available at **[Zenodo DOI to be added]**.

```bash
pip install -e ".[reference]"
```

```python
import anndata as ad
from scboost_ensemble.reference_model import ReferenceModel

model = ReferenceModel.load("ts_v2_full", device="cpu")    # or "cuda:0"
query = ad.read_h5ad("query.h5ad")    # raw counts in X; var_names are Ensembl gene IDs
res = model.project(query, donor_key="donor_id")
labels = res["label"]["xgboost"]      # one of the 167 reference cell types per cell
proba = res["proba"]["xgboost"]       # (n_cells, 167)
```

- **Genes** are matched on Ensembl ID, with any version suffix removed. Reference genes
  that the query lacks are filled with zeros. `project` stops if a query lacks more than
  10% of either HVG list; pass `allow_missing=True` to proceed anyway.
- **Counts** can also come from a layer (`counts_layer="counts"`) or from
  `raw.X` (`counts_layer="raw"`).
- **Meta-learners.** Two are saved: `xgboost`, which the manuscript reports, and
  `torch_logreg_en`. `res["compartment_label"]` gives the coarse compartment of each call.
- **Speed.** scVI surgery is the slow step: about 2 minutes per 10,000 cells on one
  A100 GPU, measured on 734,078 cells.
- **CPU and GPU.** The two give the same labels. Their xgboost probabilities can differ
  in the third decimal place: on 14,291 query cells the largest difference was 0.0036,
  with every label identical. The manuscript's results were computed on a GPU, and on a
  GPU the package reproduces them exactly.
- **`ref_embeddings/`** holds the reference cells' own embeddings. Projection does not
  need it, so it can be left out of a download.

To fit your own reference, pass raw counts in `X` and the columns `trained_label`,
`donor_id` and `batch` in `obs`:

```python
model = ReferenceModel(device="cuda:0").fit(ref)
model.set_compartments(compartment_of_each_label)    # pandas Series: label -> compartment
model.save("my_model")
```

`ReferenceModel(class_weight_beta=b)` trains every learner with class weights
w_c ∝ n_c^(−b) (`scboost_ensemble.balanced`). b = 0 reproduces the unweighted
model exactly, and b = 1 gives every class the same total weight.

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
