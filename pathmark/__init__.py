"""PathMark — watermarking for Mixture-of-Experts language models via
path-based routing constraints.

Top-level imports give convenient access to the model registry; deeper
modules are exposed through `pathmark.{models, losses, eval, ...}`.
"""
from pathmark.version import __version__
from pathmark.models import get_model_config, list_models

__all__ = [
    "__version__",
    "get_model_config",
    "list_models",
]
