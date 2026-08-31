"""
Unit tests for temperature scaling, uncertainty estimation, and abstention triggers.
"""

import numpy as np
from uncertainty.calibration import TemperatureScaler
from uncertainty.uncertainty_estimator import UncertaintyEstimator


def test_temperature_scaler():
    scaler = TemperatureScaler(temperature=1.5)
    logits = np.array([2.0, 1.0, 0.5])
    cal_probs = scaler.calibrate_probabilities(logits)
    assert np.isclose(np.sum(cal_probs), 1.0)
    assert len(cal_probs) == 3


def test_uncertainty_low():
    estimator = UncertaintyEstimator()
    # High confidence spike
    logits = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0])
    res = estimator.estimate(logits)
    assert res.uncertainty_level in ["LOW", "MEDIUM"]
    assert res.calibrated_confidence > 0.70


def test_uncertainty_high_abstention():
    estimator = UncertaintyEstimator()
    # Uniform / flat distribution -> high entropy & uncertainty
    logits = np.zeros(9)
    res = estimator.estimate(logits)
    assert res.uncertainty_level == "HIGH"
    assert res.review_required is True
    assert "Pathologist review recommended" in res.abstention_message
