# Changelog

## 0.3.0 (2026-11-06)

**Pretrained reference model.** Three new modules carry the code behind the
manuscript's pretrained scBoost, which is fitted once on Tabula Sapiens 2.0 and
applied to new datasets:

| module | contents | source (md5) |
|---|---|---|
| `reference_model` | `ReferenceModel`: fit, save, load, project | analysis code `lib/reference_model.py` (`3da9045b…`), adapted |
| `folds` | hybrid donor folds | `lib/folds.py` (`420407e3…`), byte-identical |
| `balanced` | class-balanced losses, weight ∝ n_c^(−β) | `lib/balanced.py` (`0c4ab123…`), byte-identical |

`reference_model` differs from its source in four ways:
1. It imports from the package instead of a vendored copy.
2. It records the package's own md5s in `manifest.json`.
3. It loads models saved by the analysis code, whose pickles name the classes under
   `lib.*`.
4. It moves torch tensors to the CPU while loading, so a model saved on a GPU loads
   on a machine without one.

Fitting and projection are unchanged. These checks used the manuscript's
`ts_v2_full` model and the saved embeddings of 14,291 Plaque Atlas cells:

- On a GPU, the package reproduces the saved manuscript probabilities bit for bit,
  for both meta-learners.
- On the CPU and on the GPU, the package's outputs are bit-identical to the
  analysis code's.

New optional extra: `pip install ".[reference]"`.

**Base-learner cache keys.**
- Cache directory names gain a `d{N}` input-width segment.
- A cached model trained at a different width is rejected.
- Each entry records a `content_sha` of its training matrix (`--strict_cache`).
- Pre-0.3.0 entries are reused when their width matches (`--legacy_cache`).
- Only cache naming and lookup changed. Training and prediction did not.

**`embedding_metrics`.** Pairwise CKA, linear-map R² and residuals between `.obsm`
embeddings.

## 0.2.0

The stacking engine used for every result in the scBoost manuscript (tag `v0.2.0`).
`stacking.py` has md5 `c78a0c1b2bd33ca026d0880081122c1e`. The 0.3.0 release leaves
`torch_base`, `torch_meta`, `meta_learners`, `eval`, `inputs`, `splits` and `_xgb_utils`
unchanged.
