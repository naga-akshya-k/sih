"""
Robust Feature Loader for Morphological Measurements and Vectors.
"""

import json
import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd

from .feature_schema import MorphologyFeatureVector, CaseSummaryData, MORPHOLOGY_FEATURE_KEYS

logger = logging.getLogger(__name__)


class FeatureLoader:
    """
    Validates and loads morphological features from JSON, CSV, or dictionary formats.
    """

    @classmethod
    def load_feature_vector(
        cls, source: Union[str, Path, Dict[str, Any]]
    ) -> MorphologyFeatureVector:
        """
        Loads a MorphologyFeatureVector from a JSON path or dictionary.
        """
        data: Dict[str, Any] = {}
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Feature vector file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(source, dict):
            data = source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Ensure case_id
        if "case_id" not in data:
            data["case_id"] = "unknown_case"

        # Validate and sanitize numeric fields
        cleaned_data: Dict[str, Any] = {"case_id": str(data["case_id"])}
        for key in MORPHOLOGY_FEATURE_KEYS:
            raw_val = data.get(key, 0.0)
            try:
                val = float(raw_val)
                if np.isnan(val) or np.isinf(val):
                    logger.warning(f"Field '{key}' has non-finite value ({raw_val}), setting to 0.0")
                    val = 0.0
                if "total" in key or "type_" in key:
                    cleaned_data[key] = int(val)
                else:
                    cleaned_data[key] = float(val)
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for '{key}': {raw_val}, defaulting to 0")
                cleaned_data[key] = 0

        return MorphologyFeatureVector(**cleaned_data)

    @classmethod
    def from_case_summary(
        cls, summary_source: Union[str, Path, Dict[str, Any]]
    ) -> MorphologyFeatureVector:
        """
        Derives MorphologyFeatureVector directly from case_summary.json.
        """
        if isinstance(summary_source, (str, Path)):
            with open(summary_source, "r", encoding="utf-8") as f:
                raw_summary = json.load(f)
        else:
            raw_summary = summary_source

        case_summary = CaseSummaryData(**raw_summary)
        nuclei = case_summary.nuclei
        glands = case_summary.glands

        vector_data = {
            "case_id": case_summary.case_id,
            "nuclei_total": nuclei.total,
            "nuclei_type_1": nuclei.types.get("1", 0),
            "nuclei_type_2": nuclei.types.get("2", 0),
            "nuclei_type_3": nuclei.types.get("3", 0),
            "nuclei_type_4": nuclei.types.get("4", 0),
            "nuclei_mean_area_px2": nuclei.mean_area_px2,
            "nuclei_mean_perimeter_px": nuclei.mean_perimeter_px,
            "nuclei_mean_eccentricity": nuclei.mean_eccentricity,
            "nuclei_mean_circularity": nuclei.mean_circularity,
            "glands_total": glands.total,
            "glands_mean_area_px2": glands.mean_area_pixels,
            "glands_mean_perimeter_px": glands.mean_perimeter_pixels,
            "glands_mean_width_px": glands.mean_width_pixels,
            "glands_mean_height_px": glands.mean_height_pixels,
            "glands_mean_aspect_ratio": glands.mean_aspect_ratio,
            "glands_mean_circularity": glands.mean_circularity,
        }

        return cls.load_feature_vector(vector_data)

    @classmethod
    def from_measurements(
        cls,
        case_id: str,
        nuclei_csv: Optional[Union[str, Path]] = None,
        glands_csv: Optional[Union[str, Path]] = None,
    ) -> MorphologyFeatureVector:
        """
        Constructs a feature vector directly from raw nuclei and gland measurement CSV files.
        """
        n_total = 0
        n_types = {1: 0, 2: 0, 3: 0, 4: 0}
        n_area = 0.0
        n_perim = 0.0
        n_ecc = 0.0
        n_circ = 0.0

        if nuclei_csv and Path(nuclei_csv).exists():
            ndf = pd.read_csv(nuclei_csv)
            n_total = len(ndf)
            if n_total > 0:
                if "type" in ndf.columns:
                    for t, cnt in ndf["type"].value_counts().items():
                        try:
                            n_types[int(t)] = int(cnt)
                        except (ValueError, TypeError):
                            pass
                n_area = float(ndf["area_px2"].mean()) if "area_px2" in ndf.columns else 0.0
                n_perim = float(ndf["perimeter_px"].mean()) if "perimeter_px" in ndf.columns else 0.0
                n_ecc = float(ndf["eccentricity"].dropna().mean()) if "eccentricity" in ndf.columns else 0.0
                n_circ = float(ndf["circularity"].dropna().mean()) if "circularity" in ndf.columns else 0.0

        g_total = 0
        g_area = 0.0
        g_perim = 0.0
        g_w = 0.0
        g_h = 0.0
        g_ar = 0.0
        g_circ = 0.0

        if glands_csv and Path(glands_csv).exists():
            gdf = pd.read_csv(glands_csv)
            g_total = len(gdf)
            if g_total > 0:
                g_area = float(gdf["area_pixels"].mean()) if "area_pixels" in gdf.columns else 0.0
                g_perim = float(gdf["perimeter_pixels"].mean()) if "perimeter_pixels" in gdf.columns else 0.0
                g_w = float(gdf["width_pixels"].mean()) if "width_pixels" in gdf.columns else 0.0
                g_h = float(gdf["height_pixels"].mean()) if "height_pixels" in gdf.columns else 0.0
                g_ar = float(gdf["aspect_ratio"].mean()) if "aspect_ratio" in gdf.columns else 0.0
                g_circ = float(gdf["circularity"].mean()) if "circularity" in gdf.columns else 0.0

        data = {
            "case_id": case_id,
            "nuclei_total": n_total,
            "nuclei_type_1": n_types.get(1, 0),
            "nuclei_type_2": n_types.get(2, 0),
            "nuclei_type_3": n_types.get(3, 0),
            "nuclei_type_4": n_types.get(4, 0),
            "nuclei_mean_area_px2": n_area,
            "nuclei_mean_perimeter_px": n_perim,
            "nuclei_mean_eccentricity": n_ecc,
            "nuclei_mean_circularity": n_circ,
            "glands_total": g_total,
            "glands_mean_area_px2": g_area,
            "glands_mean_perimeter_px": g_perim,
            "glands_mean_width_px": g_w,
            "glands_mean_height_px": g_h,
            "glands_mean_aspect_ratio": g_ar,
            "glands_mean_circularity": g_circ,
        }

        return cls.load_feature_vector(data)
