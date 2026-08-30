"""
Classifiers Package for COLONPATH-AI.
Provides dataset loaders, multimodal training routines, and evaluation metrics.
"""

from .tissue_classifier import TissueClassifier, get_tissue_classifier
from .dataset import ColorectalDataset, create_data_splits

__all__ = [
    "TissueClassifier",
    "get_tissue_classifier",
    "ColorectalDataset",
    "create_data_splits",
]
