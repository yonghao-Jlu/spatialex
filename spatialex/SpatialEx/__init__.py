# -*- coding: utf-8 -*-
"""
SpatialEx package for spatial transcriptomics analysis.

This package provides tools for histology-to-omics translation using
hypergraph neural networks on spatial transcriptomics data.
"""

from . import model
from . import preprocess
from . import utils

# Import main training classes
try:
    from .SpatialEx_pyG import Train_SpatialEx, Train_SpatialExP, Train_SpatialExP_Big
except ImportError as e:
    # Handle import errors gracefully for documentation generation
    import warnings
    warnings.warn(f"Could not import SpatialEx_pyG: {e}")

__version__ = "0.1.0"
__all__ = [
    "model",
    "preprocess", 
    "utils",
    "Train_SpatialEx",
    "Train_SpatialExP", 
    "Train_SpatialExP_Big"
]
