"""
Unit tests for morphology feature loader, normalizer, and fusion model.
"""

from pathlib import Path
import numpy as np
import torch
import pytest
from fusion.feature_loader import FeatureLoader
from fusion.normalization import FeatureNormalizer
from fusion.fusion_model import MultimodalFusionNet

FV_PATH = Path(__file__).resolve().parents[1] / "outputs" / "morphology" / "feature_vector.json"


def test_feature_loader():
    if not FV_PATH.exists():
        pytest.skip("Feature vector not found")
    fv = FeatureLoader.load_feature_vector(FV_PATH)
    arr = fv.to_numpy()
    assert arr.shape == (16,)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_feature_normalizer():
    dummy = np.random.randn(20, 16).astype(np.float32) * 50.0 + 10.0
    normalizer = FeatureNormalizer()
    normed = normalizer.fit_transform(dummy)
    assert normed.shape == (20, 16)
    assert np.allclose(np.mean(normed, axis=0), 0.0, atol=1e-1)


def test_multimodal_fusion_net():
    model = MultimodalFusionNet()
    model.eval()
    v = torch.randn(2, 1024)
    m = torch.randn(2, 16)
    mc_logits, bin_logits, latent = model(v, m)
    assert mc_logits.shape == (2, 9)
    assert bin_logits.shape == (2, 2)
    assert latent.shape == (2, 128)
