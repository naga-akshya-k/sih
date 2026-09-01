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
    ood_score: float = Field(default=0.0, description="Energy-based Out-of-Distribution score")
    ood_status: str = Field(default="IN_DISTRIBUTION", description="IN_DISTRIBUTION or OOD_DETECTED")
    is_ood: bool = Field(default=False, description="True if input falls outside training distribution")
    review_required: bool
    abstention_message: str


class UncertaintyEstimator:
    """
    Computes entropy, calibrated confidence, uncertainty levels, energy-based OOD detection,
    and handles automated abstention.
    """

    def __init__(
        self,
        scaler: Optional[TemperatureScaler] = None,
        uncertainty_threshold_high: float = 0.50,
        uncertainty_threshold_med: float = 0.25,
        min_confidence_threshold: float = 0.60,
        ood_energy_threshold: float = -2.5,
    ):
        self.scaler = scaler or TemperatureScaler()
        self.thresh_high = uncertainty_threshold_high
        self.thresh_med = uncertainty_threshold_med
        self.min_confidence = min_confidence_threshold
        self.ood_energy_threshold = ood_energy_threshold

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

        # 4. Energy-Based Out-Of-Distribution (OOD) Detection: E(x; T) = -T * log(sum(exp(logits / T)))
        t_val = float(self.scaler.temperature.item()) if hasattr(self.scaler, "temperature") else 1.25
        # Numerical stability via max subtraction
        max_l = np.max(logits_arr / t_val)
        exp_sum = np.sum(np.exp((logits_arr / t_val) - max_l))
        free_energy = float(-t_val * (max_l + np.log(exp_sum + eps)))
        
        # Normalize OOD score into [0, 1] range using stable sigmoid
        clipped_e = float(np.clip(-free_energy, -50.0, 50.0))
        norm_ood_score = float(1.0 / (1.0 + np.exp(-clipped_e)))
        is_ood = free_energy > self.ood_energy_threshold or not image_quality_passed
        ood_status = "OOD_DETECTED" if is_ood else "IN_DISTRIBUTION"

        # 5. Composite Uncertainty Score: (Normalized Entropy + Margin Uncertainty) / 2
        uncertainty_score = float(0.6 * norm_entropy + 0.4 * margin_uncertainty)
        if not image_quality_passed or is_ood:
            uncertainty_score = min(1.0, uncertainty_score + 0.30)

        # 6. Categorize Uncertainty Level
        if is_ood and not image_quality_passed:
            level = "HIGH"
            review_required = True
            msg = "Image quality compromised (blur/contrast). Pathologist review mandatory."
        elif is_ood:
            level = "HIGH"
            review_required = True
            msg = "OOD / unsupported tissue input detected. Automated prediction abstained."
        elif uncertainty_score >= self.thresh_high or cal_conf < self.min_confidence:
            level = "HIGH"
            review_required = True
            if cal_conf < self.min_confidence:
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
            ood_score=norm_ood_score,
            ood_status=ood_status,
            is_ood=is_ood,
            review_required=review_required,
            abstention_message=msg,
        )
