"""scboost-ensemble — donor-aware stacking ensemble for scRNA-seq cell-type classification."""

__version__ = "0.2.0"

from .stacking import StackingEnsembleDonorMetaCached
from .inputs import build_latents_h5ad, build_concat_h5ad, build_probs_h5ad
from . import plots
from . import splits
from . import eval as eval_  # avoid shadowing the builtin

__all__ = [
    "__version__",
    "StackingEnsembleDonorMetaCached",
    "build_latents_h5ad",
    "build_concat_h5ad",
    "build_probs_h5ad",
    "plots",
    "splits",
    "eval_",
]
