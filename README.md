# scboost-ensemble

scboost-ensemble is a stacking ensemble for cell-type classification in single-cell
RNA-seq data. It takes several different cell embeddings as input, such as scVI,
Harmony and Scanorama. It trains one classifier on each embedding, and a meta-learner
then combines their predictions into a single cell-type call. This package is the
software behind scBoost.

## How it works

For each embedding, a base classifier (XGBoost or a small neural network) predicts
the probability of every cell type for every cell. A meta-learner then learns how to
combine these probabilities. We implement two features that improve on standard
stacking:

1. **Five donor-aware meta-learners.** The training cells are split into five folds,
   and with `cv_scheme="donor"` all cells of a donor stay in the same fold.
   Meta-learner *k* is trained on the out-of-fold predictions for the cells outside
   fold *k*, and the cells in fold *k* are used only to decide when to stop training.
   Because fold *k* holds donors the meta-learner has not seen, training stops at a
   point that carries over to new donors rather than one tuned to the training donors.
   At prediction time, the outputs of the five meta-learners are averaged.
2. **Cached base classifiers.** Each base classifier, one per embedding and fold, is
   saved to disk under a name derived from everything that determines its training:
   the embedding, the learner and its settings, the fold, the cell types and the
   training cells. A later run that needs the same classifier loads it instead of
   training it again. In experiments that compare many combinations of embeddings,
   where the same base classifier appears in many runs, this makes the runs about
   5 to 10 times faster.

## Installation

```bash
git clone https://github.com/kanghelen/scboost-ensemble
cd scboost-ensemble
pip install -e .
```

Optional features are installed as extras, for example `pip install -e ".[torch,umap]"`:

| extra | installs | needed for |
|---|---|---|
| `torch` | PyTorch | the `mlp` and `logistic` base learners and the `torch_*` meta-learners |
| `umap` | umap-learn | UMAP views in `plots.plot_oof_meta_features` |
| `reference` | PyTorch, scvi-tools, scikit-misc, h5py, pyarrow | the pretrained reference model (see below) |
| `dev` | pytest, pytest-cov, ruff, jupyter | running the tests and the example notebooks |

scboost-ensemble requires Python 3.10 or newer. A GPU is optional but recommended;
XGBoost uses it when you pass `device="cuda:0"` in Python or `--gpu 0` on the
command line. PyTorch is imported only when a learner needs it, so the package installs
and runs without it.

## Quick start in Python

The input is an AnnData object whose `.obsm` holds one dense matrix per embedding. Its
`.obs` holds the cell-type labels, a donor (or batch) column, and a `split` column with
the values `train` and `test`. The class does not split the data itself: you pass the
training cells to `fit` and the cells to classify to `predict`.

```python
import scanpy as sc
from scboost_ensemble import StackingEnsembleDonorMetaCached

adata = sc.read_h5ad("my_data.h5ad")
train = adata[adata.obs["split"] == "train"].copy()
test = adata[adata.obs["split"] == "test"].copy()

stacker = StackingEnsembleDonorMetaCached(
    obsm_keys=["scVI", "Harmony", "Scanorama"],  # one base classifier per embedding
    label_key="cell_type",
    batch_key="donor_id",       # the folds are grouped by this column
    cv_scheme="donor",          # keep all cells of a donor in one fold
    n_folds=5,
    use_batch_feature=False,    # see the note below
    device="cuda:0", n_gpus=1,
    base_cache_dir="./cache",   # reuse base classifiers across runs
)
stacker.fit(train)
labels = stacker.predict(test)
proba = stacker.predict_proba(test)

print(stacker.get_method_importance())
```

`get_method_importance()` reports how much each embedding contributes to the
meta-learners' decisions, as a share of their total feature importance averaged over
the five meta-learners.

**Batch feature.** By default, the meta-learner also receives a one-hot encoding of
each cell's batch. A batch that never appears in training is encoded as all zeros,
which is misleading for cells from new donors. When you classify cells from held-out
donors, please set `use_batch_feature=False` (`--no_batch_feature` on the command line).
All results in the scBoost manuscript were computed this way.

## Quick start on the command line

```bash
scboost fit \
    --input my_data.h5ad \
    --sample_name run01 \
    --methods scVI Harmony Scanorama \
    --out_dir results/ \
    --split_type donor_id --batch_key donor_id --cv_scheme donor \
    --no_batch_feature \
    --gpu 0 --n_gpus 1 \
    --base_cache_dir ./cache
```

`--methods` lists the `.obsm` keys to use. With `--split_type donor_id`, the cells of
one donor become the test set. That donor is the one given by `--holdout_value` or,
if none is given, the donor whose cell count is closest to `--test_size` of all cells
(default 0.1). With `--split_type stratified` (the default), a random `--test_size` share
of the cells is held out instead, stratified by cell type and batch. Test cells whose cell type never occurs in
the training set are removed before scoring, and the number removed is printed.

The run writes these files to `--out_dir`:

| file | contents |
|---|---|
| `<sample>_classification_report.tsv` | precision, recall, F1 and support for each cell type, with overall accuracy and macro and weighted averages |
| `<sample>_method_importance.tsv` | the contribution of each embedding to the meta-learners |
| `<sample>_training_summary.csv` | the best training iteration of each learner in each fold |
| `<sample>_runtime.tsv` | the time spent in each phase |
| `<sample>_split.tsv` | whether each cell was in the training or the test set |

## Choosing the base and meta-learners

Both levels of the stack can be swapped, and the choice can matter. On
high-dimensional embeddings, such as foundation-model embeddings with 512 to 2,048
dimensions, an MLP base learner can beat XGBoost by 2 percentage points or more. On
100-dimensional batch-corrected embeddings, the two are close, and which one is better
depends on the dataset. Please consider trying both MLP and XGBoost on your data.

| option | choices | notes |
|---|---|---|
| `--base_algo` | `xgboost` (default), `mlp`, `logistic` | `mlp` is a neural network with two hidden layers; set their sizes with `--base_hidden` (default 256 and 128) and the dropout with `--base_dropout` (default 0.3). `logistic` is multinomial logistic regression. Both need PyTorch. |
| `--meta_algo` | `xgboost` (default), `torch_logreg_en`, `torch_elasticnet_reg`, `logreg_en`, `elasticnet_reg` | `torch_logreg_en` and `torch_elasticnet_reg` are elastic-net models that run on the GPU and need PyTorch. `logreg_en` and `elasticnet_reg` are CPU-only scikit-learn versions. |

In Python, the same choices are the `base_algo` and `meta_algo` arguments. For example:

```bash
scboost fit --input data.h5ad --sample_name run02 --methods scVI Harmony Scanorama \
    --base_algo mlp --meta_algo xgboost \
    --split_type donor_id --batch_key donor_id --cv_scheme donor --no_batch_feature \
    --out_dir out/
```

**Reproducibility.** Given the same cached base classifiers, the XGBoost meta-learner
gives bit-identical results. The `torch_*` meta-learners set a random seed but do not
force deterministic GPU arithmetic. Their results are therefore reproducible on one
machine, but can differ by about 0.2 percentage points between machines, because GPU
sums are computed in a different order. If you need identical results on different
machines, please use the XGBoost meta-learner.

## The base-classifier cache

When `base_cache_dir` is set, each base classifier is saved in its own directory. The
directory name records everything that determines the training, for example:

```
Geoksketch__xgboost__fold0__d100__donor5__es10__p<params>__c<classes>__t<train_ids>
```

From left to right, the name gives:
- the embedding;
- the base learner;
- the fold;
- the number of input dimensions (`d100`, added in version 0.3.0);
- the fold scheme and the number of folds;
- the early-stopping setting;
- short hashes of the learner's settings, of the set of cell types and of the training cells.

The width in the name keeps a 100-dimensional and a 50-dimensional embedding apart
when they share an `.obsm` name. Two further checks guard against reusing the wrong
classifier:

- **Width check.** Before a cached classifier is used, the number of dimensions it was
  trained on is compared with the current embedding. If they differ, the classifier is
  retrained rather than failing at prediction time.
- **Content check.** Each cache entry records a fingerprint (`content_sha`) of the matrix
  it was trained on. If an embedding changes but keeps both its name and its width, the
  run prints a warning. With `--strict_cache` (`strict_cache=True` in Python), the
  entry is treated as missing and the classifier is retrained instead. We recommend
  `--strict_cache` for any run whose results you will publish, because a warning is
  easy to miss in a long log.

The simplest way to avoid these problems is to give each version of an embedding its
own name in `.obsm`, such as `Python_PCA_50d` and `Python_PCA_100d`, rather than
overwriting an embedding under an existing name.

Cache entries written before version 0.3.0 have no width in their names. By default
(`--legacy_cache adopt`), such an entry is reused when the classifier inside it has
the right width, and a link to it is added under the new name. These reuses count as
cache hits and are also counted separately in `cache_stats_["legacy_adopted"]`. With
`--legacy_cache ignore`, the classifier is retrained instead. Old entries have no
fingerprint, so a change in content at the same width cannot be detected. If that
matters, please use a new cache directory with `--strict_cache`.

## Pretrained reference model

`scboost_ensemble.reference_model`, new in version 0.3.0, trains scBoost once on a
labelled reference dataset and saves every trained component. It can then label new
query cells without retraining anything. Only three of the usual embeddings can place
new cells in a fixed space, so the reference model uses those three:

- **PCA**: 100 components. They are computed on the 3,000 most variable genes of the
  reference, after each cell is normalised to 10,000 counts and log-transformed.
- **NMF**: 100 components on the same matrix. The gene programs are kept fixed, and each
  query cell is expressed in terms of them.
- **scVI**: 30 latent dimensions, trained on the 2,000 most variable genes of the
  reference. Query cells are added by scArches surgery, which treats each query donor as
  a new batch.

Each embedding has five base classifiers, one per fold, so the model holds
3 (embeddings) × 5 (folds) = 15 MLP base classifiers and five meta-learners. The folds
keep the cells of each donor together. The exception is cell types found in only one
donor: their cells are spread across all five folds (`scboost_ensemble.folds`).

A model trained on Tabula Sapiens 2.0 (637,267 cells, 167 cell types) is available at
**[Zenodo DOI to be added]**. To use it:

```bash
pip install -e ".[reference]"
```

```python
import anndata as ad
from scboost_ensemble.reference_model import ReferenceModel

model = ReferenceModel.load("ts_v2_full", device="cpu")   # or device="cuda:0"
query = ad.read_h5ad("query.h5ad")    # raw counts in X; var_names are Ensembl gene IDs
res = model.project(query, donor_key="donor_id")
labels = res["label"]["xgboost"]      # one of the 167 reference cell types per cell
proba = res["proba"]["xgboost"]       # probabilities, shape (n_cells, 167)
```

- **Genes.** Query genes are matched to the reference by Ensembl ID, ignoring version
  suffixes such as `.5`. Reference genes that the query lacks are set to zero. If the
  query lacks more than 10% of either list of variable genes, `project` stops with an
  error; pass `allow_missing=True` to continue anyway.
- **Counts.** `project` reads raw counts from `X` by default. Use `counts_layer="counts"`
  to read a layer, or `counts_layer="raw"` to read `raw.X`.
- **Outputs.** `project` returns the predictions of two meta-learners: `xgboost`, which
  the scBoost manuscript reports, and `torch_logreg_en`. `res["compartment_label"]` gives
  each cell's broad compartment, such as Immune or Endothelium.
- **Speed.** The scVI surgery takes most of the time, about 2 minutes per 10,000 cells
  on one A100 GPU.
- **CPU or GPU.** Both give the same labels. The probabilities from the XGBoost
  meta-learner can differ slightly: on 14,291 query cells, the largest difference was
  0.0036, and every label agreed. The manuscript's results were computed on an A100 GPU,
  and rerunning the same cells on the same type of GPU reproduces them exactly. A
  different device or a different number of cells changes the floating-point arithmetic
  slightly.
- **Download size.** The folder `ref_embeddings/` holds the embeddings of the reference
  cells themselves. `project` does not use it, so you can leave it out of the download.

To train your own reference model, pass an AnnData object with raw counts in `X` and
three columns in `.obs`: `trained_label` (the cell type), `donor_id` and `batch`.

```python
model = ReferenceModel(device="cuda:0").fit(ref)
model.set_compartments(compartment_of_each_label)   # a pandas Series: cell type -> compartment
model.save("my_model")
```

`set_compartments` assigns each cell type to a compartment, so that predictions can
also be summarised at the compartment level.

`ReferenceModel(class_weight_beta=b)` weights each training cell in proportion to
n_c^(−b), where n_c is the number of training cells of its type
(`scboost_ensemble.balanced`). With b = 0, the model is identical to the unweighted
one. With b = 1, every cell type carries the same total weight.

## Building inputs from another annotator

If another tool, such as popV, has saved its embeddings or probabilities in an `.npz`
file, these commands turn them into input files for scboost-ensemble:

```bash
# Put each embedding in its own .obsm entry.
scboost build-latents \
    --npz_path popv_latents.npz \
    --reference_h5ad source.h5ad \
    --out outputs/scboost_input.h5ad

# Also add each pair of embeddings joined side by side, and all of them joined
# together (useful for ablation studies).
scboost build-concat \
    --npz_path popv_latents.npz \
    --reference_h5ad source.h5ad \
    --out outputs/concat_input.h5ad

# Put each classifier's probability matrix in .obsm, so that the meta-learner can
# combine the classifiers directly.
scboost build-from-probs \
    --npz_path popv_probs.npz \
    --reference_h5ad source.h5ad \
    --out outputs/probs_input.h5ad \
    --prefix popV_prob_
```

## Plots

The `plots` module draws diagnostic figures from a fitted model:

```python
from scboost_ensemble import plots

plots.plot_method_importance(stacker)      # contribution of each embedding
plots.plot_training_summary(stacker)       # best iteration of each learner
plots.plot_classification_report("results/run01_classification_report.tsv")
plots.plot_oof_meta_features(stacker, train, projection="umap")   # needs the umap extra
```

The notebook `examples/04_plotting.ipynb` walks through each of them.

## Example notebooks

| notebook | what it shows |
|---|---|
| `examples/01_basic_stacking.ipynb` | stacking two embeddings on a small dataset |
| `examples/02_concat_embeddings.ipynb` | using `build_concat_h5ad` for ablation studies |
| `examples/03_prob_integration.ipynb` | a meta-learner trained directly on probabilities from any annotator |
| `examples/04_plotting.ipynb` | the diagnostic plots, including a projection of the out-of-fold meta-features |

## License

scboost-ensemble is released under the MIT License (see `LICENSE`).
