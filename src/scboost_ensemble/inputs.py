"""
Input-building helpers for scboost-ensemble.

Build minimal AnnData objects with ``.obs`` (cell metadata) + ``.obsm`` (one or
more embedding matrices) suitable for ``StackingEnsembleDonorMetaCached.fit()``.

Three variants:

  build_latents_h5ad   — from per-method latent embeddings (1-to-1 in obsm)
  build_concat_h5ad    — concatenate combinations of latents (pairwise + all)
  build_probs_h5ad     — from per-method probability matrices (n × n_classes)

Inputs typically come from a third-party annotator (e.g. popV) saved as a
compressed ``.npz`` with one ``cell_ids`` array + one matrix per method.
"""

from __future__ import annotations
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence, Dict

import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_rows_to_ref(cell_ids: np.ndarray, ref_h5ad: str) -> tuple:
    """Look up the rows of `ref_h5ad` corresponding to `cell_ids`."""
    ref = ad.read_h5ad(ref_h5ad, backed="r")
    ref_index = {cid: i for i, cid in enumerate(ref.obs.index.astype(str))}
    missing = [cid for cid in cell_ids if cid not in ref_index]
    if missing:
        raise RuntimeError(f"{len(missing)} cell IDs in source npz not in reference h5ad. "
                           f"First missing: {missing[:5]}")
    rows = np.array([ref_index[cid] for cid in cell_ids], dtype=np.int64)
    ref_obs = ref.obs.iloc[rows].copy()
    ref.file.close()
    return ref_obs, rows


def _maybe_add_split(ref_obs: pd.DataFrame, cell_ids: np.ndarray,
                      split_h5ad: Optional[str]) -> pd.DataFrame:
    """If a `split_h5ad` is given (with obs['_dataset'] in {ref, query}),
    add a 'split' column ('train' for ref, 'test' for query)."""
    if split_h5ad is None:
        return ref_obs
    split_ad = ad.read_h5ad(split_h5ad, backed="r")
    split_index = {cid: i for i, cid in enumerate(split_ad.obs.index.astype(str))}
    labels = []
    for cid in cell_ids:
        i = split_index.get(cid)
        ds = split_ad.obs["_dataset"].iloc[i] if i is not None else "unknown"
        labels.append("train" if ds == "ref" else "test")
    split_ad.file.close()
    ref_obs["split"] = labels
    return ref_obs


def _build_minimal_adata(cell_ids: np.ndarray, ref_obs: pd.DataFrame,
                          obsm: Dict[str, np.ndarray]) -> ad.AnnData:
    """Wrap obs + obsm into a minimal AnnData with a 1-column dummy .X."""
    X = sp.csr_matrix((len(cell_ids), 1), dtype=np.float32)
    adata = ad.AnnData(X=X,
                       obs=ref_obs.reset_index(drop=False).set_index("index"),
                       obsm={k: v.astype(np.float32) for k, v in obsm.items()})
    adata.obs.index = cell_ids
    adata.obs.index.name = None
    return adata


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_latents_h5ad(
    npz_path: str,
    reference_h5ad: str,
    out_path: str,
    *,
    split_h5ad: Optional[str] = None,
    keys: Optional[Sequence[str]] = None,
) -> ad.AnnData:
    """Build an h5ad with each latent matrix from `npz_path` as a separate obsm entry.

    Parameters
    ----------
    npz_path : str
        .npz with 'cell_ids' array + one (n, d) matrix per latent method.
    reference_h5ad : str
        Source h5ad providing cell metadata (.obs columns).
    out_path : str
        Where to write the minimal h5ad.
    split_h5ad : str, optional
        h5ad with obs['_dataset'] in {ref, query} to populate `.obs['split']`.
    keys : sequence of str, optional
        Subset of npz keys to include (default: all except 'cell_ids'/'class_labels').
    """
    z = np.load(npz_path, allow_pickle=False)
    cell_ids = z["cell_ids"].astype(str)
    if keys is None:
        keys = [k for k in z.files if k not in ("cell_ids", "class_labels")]
    obsm = {k: np.asarray(z[k]) for k in keys}
    ref_obs, _ = _align_rows_to_ref(cell_ids, reference_h5ad)
    ref_obs = _maybe_add_split(ref_obs, cell_ids, split_h5ad)
    adata = _build_minimal_adata(cell_ids, ref_obs, obsm)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"Saved → {out_path}  ({round(Path(out_path).stat().st_size/1024**2,1)} MB)")
    return adata


def build_concat_h5ad(
    npz_path: str,
    reference_h5ad: str,
    out_path: str,
    *,
    split_h5ad: Optional[str] = None,
    base_keys: Optional[Sequence[str]] = None,
    pair_prefix: str = "concat",
    all_key: str = "concat_all",
    include_pairs: bool = True,
    include_all: bool = True,
) -> ad.AnnData:
    """Build an h5ad with all pairwise concatenations + optional all-N concatenation
    of the latent matrices in `npz_path`.

    For N base latents, produces C(N, 2) pair concats (if `include_pairs`) and
    optionally one all-N concat. Concat axis is horizontal; obsm keys are
    {pair_prefix}_{m1}_{m2} and {all_key}.
    """
    z = np.load(npz_path, allow_pickle=False)
    cell_ids = z["cell_ids"].astype(str)
    if base_keys is None:
        base_keys = [k for k in z.files if k not in ("cell_ids", "class_labels")]
    base_keys = list(base_keys)
    latents = {k: np.asarray(z[k]) for k in base_keys}

    obsm = {}
    if include_pairs:
        for m1, m2 in combinations(base_keys, 2):
            obsm[f"{pair_prefix}_{m1}_{m2}"] = np.hstack([latents[m1], latents[m2]])
    if include_all and len(base_keys) >= 2:
        obsm[all_key] = np.hstack([latents[k] for k in base_keys])

    ref_obs, _ = _align_rows_to_ref(cell_ids, reference_h5ad)
    ref_obs = _maybe_add_split(ref_obs, cell_ids, split_h5ad)
    adata = _build_minimal_adata(cell_ids, ref_obs, obsm)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"Saved → {out_path}  ({round(Path(out_path).stat().st_size/1024**2,1)} MB)  "
          f"obsm keys: {list(adata.obsm.keys())}")
    return adata


def build_probs_h5ad(
    npz_path: str,
    reference_h5ad: str,
    out_path: str,
    *,
    split_h5ad: Optional[str] = None,
    keys: Optional[Sequence[str]] = None,
    prefix: Optional[str] = None,
) -> ad.AnnData:
    """Build an h5ad where each method's per-class probability matrix becomes
    one entry in .obsm.

    Parameters
    ----------
    npz_path : str
        .npz with 'cell_ids' + one (n, n_classes) matrix per method.
    keys : sequence of str, optional
        Subset of npz keys to include.
    prefix : str, optional
        If set, only include keys starting with this prefix (e.g. 'popV_prob_').
    """
    z = np.load(npz_path, allow_pickle=False)
    cell_ids = z["cell_ids"].astype(str)
    if keys is None:
        keys = [k for k in z.files if k not in ("cell_ids", "class_labels")]
    if prefix is not None:
        keys = [k for k in keys if k.startswith(prefix)]
    obsm = {k: np.asarray(z[k]) for k in keys}
    ref_obs, _ = _align_rows_to_ref(cell_ids, reference_h5ad)
    ref_obs = _maybe_add_split(ref_obs, cell_ids, split_h5ad)
    adata = _build_minimal_adata(cell_ids, ref_obs, obsm)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"Saved → {out_path}  ({round(Path(out_path).stat().st_size/1024**2,1)} MB)  "
          f"obsm keys: {list(adata.obsm.keys())}")
    return adata
