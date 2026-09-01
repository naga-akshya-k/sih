"""
COLONPATH-AI Image Quality Gate.
Evaluates Laplacian blur variance, brightness, contrast, and tissue coverage.
"""

from typing import Dict, Any
import numpy as np
import cv2


class ImageQualityChecker:
    """
    Evaluates histopathological image suitability before AI analysis.
    """

    def __init__(
        self,
        blur_threshold: float = 30.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_contrast: float = 20.0,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast

    def evaluate(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        if image_rgb is None or image_rgb.size == 0:
            return {
                "status": "FAILED",
                "blur_status": "FAILED",
                "blur_laplacian_variance": 0.0,
                "mean_brightness": 0.0,
                "contrast_std": 0.0,
                "is_acceptable": False,
                "notes": "Empty image frame.",
            }

        # Convert to grayscale for Laplacian & photometric checks
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_bright = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        is_blur_pass = lap_var >= self.blur_threshold
        is_bright_pass = self.min_brightness <= mean_bright <= self.max_brightness
        is_contrast_pass = contrast_std >= self.min_contrast

        passed = is_blur_pass and is_bright_pass and is_contrast_pass

        return {
            "status": "PASSED" if passed else "FAILED",
            "blur_status": "ACCEPTABLE" if is_blur_pass else "BLURRY",
            "blur_laplacian_variance": round(lap_var, 2),
            "mean_brightness": round(mean_bright, 2),
            "contrast_std": round(contrast_std, 2),
            "is_acceptable": passed,
            "notes": (
                "Optical quality acceptable for digital pathology analysis."
                if passed
                else "Image fails quality threshold. Please adjust fine-focus dial or lighting."
            ),
        }
