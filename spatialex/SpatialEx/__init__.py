"""
SpatialEx Package

A comprehensive package for spatial transcriptomics analysis using graph neural networks.
"""

# Import main classes and functions
from .SpatialEx_pyG import Train_SpatialEx, Train_SpatialExP
from .model import Model, Regression, Model_vanilla, Predictor_spot
from .utils import (
    create_optimizer, 
    Compute_metrics, 
    create_ImageEncoder,
    structural_similarity_on_graph_data
)
from . import preprocess as pp

# Version information
__version__ = "1.0.0"

# Package level imports
__all__ = [
    "Train_SpatialEx",
    "Train_SpatialExP", 
    "Model",
    "Regression",
    "Model_vanilla",
    "Predictor_spot",
    "create_optimizer",
    "Compute_metrics",
    "create_ImageEncoder",
    "structural_similarity_on_graph_data",
    "pp"
]
