"""
Feature Schemas and Data Models for Morphology Integration.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import numpy as np


class NuclearTypeCounts(BaseModel):
    epithelial: int = Field(default=0, alias="1")
    inflammatory: int = Field(default=0, alias="2")
    spindle_shaped: int = Field(default=0, alias="3")
    miscellaneous: int = Field(default=0, alias="4")


class NuclearSummary(BaseModel):
    total: int = 0
    types: Dict[str, int] = Field(default_factory=dict)
    mean_area_px2: float = 0.0
    mean_perimeter_px: float = 0.0
    mean_eccentricity: float = 0.0
    mean_circularity: float = 0.0


class GlandSummary(BaseModel):
    total: int = 0
    mean_area_pixels: float = 0.0
    mean_perimeter_pixels: float = 0.0
    mean_width_pixels: float = 0.0
    mean_height_pixels: float = 0.0
    mean_aspect_ratio: float = 0.0
    mean_circularity: float = 0.0


class CaseSummaryData(BaseModel):
    case_id: str
    nuclei: NuclearSummary
    glands: GlandSummary


MORPHOLOGY_FEATURE_KEYS = [
    "nuclei_total",
    "nuclei_type_1",
    "nuclei_type_2",
    "nuclei_type_3",
    "nuclei_type_4",
    "nuclei_mean_area_px2",
    "nuclei_mean_perimeter_px",
    "nuclei_mean_eccentricity",
    "nuclei_mean_circularity",
    "glands_total",
    "glands_mean_area_px2",
    "glands_mean_perimeter_px",
    "glands_mean_width_px",
    "glands_mean_height_px",
    "glands_mean_aspect_ratio",
    "glands_mean_circularity",
]


class MorphologyFeatureVector(BaseModel):
    case_id: str
    nuclei_total: int = 0
    nuclei_type_1: int = 0
    nuclei_type_2: int = 0
    nuclei_type_3: int = 0
    nuclei_type_4: int = 0
    nuclei_mean_area_px2: float = 0.0
    nuclei_mean_perimeter_px: float = 0.0
    nuclei_mean_eccentricity: float = 0.0
    nuclei_mean_circularity: float = 0.0
    glands_total: int = 0
    glands_mean_area_px2: float = 0.0
    glands_mean_perimeter_px: float = 0.0
    glands_mean_width_px: float = 0.0
    glands_mean_height_px: float = 0.0
    glands_mean_aspect_ratio: float = 0.0
    glands_mean_circularity: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """
        Converts the 16 morphological parameters into a clean float32 numpy array.
        """
        vals = [
            float(self.nuclei_total),
            float(self.nuclei_type_1),
            float(self.nuclei_type_2),
            float(self.nuclei_type_3),
            float(self.nuclei_type_4),
            float(self.nuclei_mean_area_px2),
            float(self.nuclei_mean_perimeter_px),
            float(self.nuclei_mean_eccentricity),
            float(self.nuclei_mean_circularity),
            float(self.glands_total),
            float(self.glands_mean_area_px2),
            float(self.glands_mean_perimeter_px),
            float(self.glands_mean_width_px),
            float(self.glands_mean_height_px),
            float(self.glands_mean_aspect_ratio),
            float(self.glands_mean_circularity),
        ]
        arr = np.array(vals, dtype=np.float32)
        # Sanitize NaNs or Infs
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e5, neginf=-1e5)
        return arr
