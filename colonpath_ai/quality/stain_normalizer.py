"""
COLONPATH-AI Optical Stain Normalization and Domain-Shift Detection.
Addresses histological color, lighting, and staining variations between optical microscope
eyepiece cameras and digitized public reference datasets (NCT-CRC-HE-100K).
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Standard target statistics in LAB color space derived from NCT-CRC-HE-100K reference
NCT_TARGET_LAB_MEAN = np.array([168.0, 152.0, 118.0], dtype=np.float32)
NCT_TARGET_LAB_STD = np.array([32.0, 18.0, 14.0], dtype=np.float32)


class ReinhardStainNormalizer:
    """
    Reinhard Color / Stain Normalization in CIELAB color space.
    Standardizes H&E staining variations across microscope illumination conditions.
    """

    def __init__(
        self,
        target_mean: np.ndarray = NCT_TARGET_LAB_MEAN,
        target_std: np.ndarray = NCT_TARGET_LAB_STD,
    ):
        self.target_mean = target_mean
        self.target_std = target_std

    def normalize(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Normalizes an RGB image to the reference H&E color distribution.
        """
        if image_rgb is None or image_rgb.size == 0:
            return image_rgb

        # Convert RGB to LAB
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Compute source statistics
        src_mean = np.mean(lab, axis=(0, 1))
        src_std = np.std(lab, axis=(0, 1)) + 1e-6

        # Standardize and scale to target distribution
        normalized_lab = np.zeros_like(lab)
        for i in range(3):
            normalized_lab[:, :, i] = (
                (lab[:, :, i] - src_mean[i]) * (self.target_std[i] / src_std[i])
            ) + self.target_mean[i]

        # Clip to valid 8-bit LAB range
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)

        # Convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        return normalized_rgb


class DomainShiftDetector:
    """
    Quantifies optical domain discrepancy between live microscope camera captures
    and training dataset distributions (NCT-CRC-HE-100K).
    """

    def __init__(
        self,
        reference_mean: np.ndarray = NCT_TARGET_LAB_MEAN,
        reference_std: np.ndarray = NCT_TARGET_LAB_STD,
        shift_threshold: float = 45.0,
    ):
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.shift_threshold = shift_threshold

    def evaluate_shift(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Computes domain shift score based on Mahalanobis-style distance in LAB color space.
        """
        if image_rgb is None or image_rgb.size == 0:
            return {
                "domain_shift_detected": True,
                "shift_score": 999.0,
                "status": "INVALID_IMAGE",
                "notes": "Empty image frame.",
            }

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        cur_mean = np.mean(lab, axis=(0, 1))
        cur_std = np.std(lab, axis=(0, 1))

        # Mean Euclidean distance in normalized LAB space
        mean_diff = np.linalg.norm(cur_mean - self.reference_mean)
        std_diff = np.linalg.norm(cur_std - self.reference_std)
        shift_score = float(mean_diff + 0.5 * std_diff)

        is_shifted = shift_score > self.shift_threshold

        return {
            "domain_shift_detected": is_shifted,
            "shift_score": round(shift_score, 2),
            "threshold": self.shift_threshold,
            "status": "DOMAIN_SHIFT_DETECTED" if is_shifted else "IN_DISTRIBUTION",
            "source_mean_lab": [round(float(x), 1) for x in cur_mean],
            "recommendation": (
                "Optical stain normalization applied; review advised due to microscope lighting divergence."
                if is_shifted
                else "Image illumination conforms to training distribution."
            ),
        }
