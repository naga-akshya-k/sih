"""
Multimodal Feature Fusion Module for COLONPATH-AI.
Combines Digepath visual foundation embeddings with structured morphology measurements.
"""

from .feature_schema import MorphologyFeatureVector, CaseSummaryData
from .feature_loader import FeatureLoader
from .normalization import FeatureNormalizer
from .fusion_model import MultimodalFusionNet

__all__ = [
    "MorphologyFeatureVector",
    "CaseSummaryData",
    "FeatureLoader",
    "FeatureNormalizer",
    "MultimodalFusionNet",
]
