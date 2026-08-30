"""
Uncertainty Estimation and Decision Abstention Engine.
"""

from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np

from .calibration import TemperatureScaler
from fusion.fusion_model import NUM_CLASSES


class UncertaintyResult(BaseModel):
    raw_confidence: float
    calibrated_confidence: float
    entropy: float
    normalized_entropy: float
    uncertainty_score: float
    uncertainty_level: str  # "LOW", "MEDIUM", "HIGH"
    review_required: bool
    abstention_message: str


class UncertaintyEstimator:
    """
    Computes entropy, calibrated confidence, uncertainty levels, and handles automated abstention.
    """

    def __init__(
        self,
        scaler: Optional[TemperatureScaler] = None,
        uncertainty_threshold_high: float = 0.50,
        uncertainty_threshold_med: float = 0.25,
        min_confidence_threshold: float = 0.60,
    ):
        self.scaler = scaler or TemperatureScaler()
        self.thresh_high = uncertainty_threshold_high
        self.thresh_med = uncertainty_threshold_med
        self.min_confidence = min_confidence_threshold

    def estimate(
        self,
        logits: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        image_quality_passed: bool = True,
    ) -> UncertaintyResult:
        """
        Estimates comprehensive uncertainty and determines if pathologist review is mandated.
        """
        logits_arr = np.asarray(logits, dtype=np.float32).ravel()

        # 1. Calibrate probabilities via Temperature Scaling
        cal_probs = self.scaler.calibrate_probabilities(logits_arr)
        sorted_cal_probs = np.sort(cal_probs)[::-1]

        raw_conf = float(np.max(probabilities)) if probabilities is not None else float(sorted_cal_probs[0])
        cal_conf = float(sorted_cal_probs[0])
        second_conf = float(sorted_cal_probs[1]) if len(sorted_cal_probs) > 1 else 0.0

        # 2. Entropy computation: H(p) = - sum(p * log(p))
        eps = 1e-10
        entropy = float(-np.sum(cal_probs * np.log(cal_probs + eps)))
        max_entropy = float(np.log(NUM_CLASSES))
        norm_entropy = float(entropy / max_entropy)

        # 3. Margin Uncertainty: Margin = 1 - (P_top1 - P_top2)
        margin_uncertainty = float(1.0 - (cal_conf - second_conf))

        # 4. Composite Uncertainty Score: (Normalized Entropy + Margin Uncertainty) / 2
        uncertainty_score = float(0.6 * norm_entropy + 0.4 * margin_uncertainty)
        if not image_quality_passed:
            uncertainty_score = min(1.0, uncertainty_score + 0.30)

        # 5. Categorize Uncertainty Level
        if uncertainty_score >= self.thresh_high or cal_conf < self.min_confidence or not image_quality_passed:
            level = "HIGH"
            review_required = True
            if not image_quality_passed:
                msg = "Image quality compromised (blur/contrast). Pathologist review mandatory."
            elif cal_conf < self.min_confidence:
                msg = "AI confidence insufficient. Pathologist review recommended."
            else:
                msg = "High model uncertainty detected. Pathologist review recommended."
        elif uncertainty_score >= self.thresh_med:
            level = "MEDIUM"
            review_required = True
            msg = "Moderate model uncertainty. Pathologist review advised."
        else:
            level = "LOW"
            review_required = False
            msg = "AI-assisted classification ready for review."

        return UncertaintyResult(
            raw_confidence=raw_conf,
            calibrated_confidence=cal_conf,
            entropy=entropy,
            normalized_entropy=norm_entropy,
            uncertainty_score=uncertainty_score,
            uncertainty_level=level,
            review_required=review_required,
            abstention_message=msg,
        )
