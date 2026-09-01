"""
Evidence and Case Result Builder.
Assembles verifiable structured computational outputs into evidence.json and case_result.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone

from fusion.feature_schema import MorphologyFeatureVector, CaseSummaryData
from uncertainty.uncertainty_estimator import UncertaintyResult
from agreement.agreement_engine import AgreementResult
from reference.reference_matcher import ReferenceComparisonResult
from regions.region_analyzer import RegionItem


class EvidenceBuilder:
    """
    Constructs deterministic case_result.json and evidence.json files.
    """

    @classmethod
    def build_case_result(
        cls,
        case_id: str,
        image_quality: Dict[str, Any],
        digepath_meta: Dict[str, Any],
        prediction_result: Dict[str, Any],
        uncertainty: UncertaintyResult,
        model_agreement: AgreementResult,
        morphology: MorphologyFeatureVector,
        reference_result: ReferenceComparisonResult,
        priority_regions: List[RegionItem],
        visualizations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Creates the standardized case_result.json structure.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Format Nuclear Evidence
        nuclear_evidence = {
            "total_count": morphology.nuclei_total,
            "type_counts": {
                "epithelial": morphology.nuclei_type_1,
                "inflammatory": morphology.nuclei_type_2,
                "spindle_shaped": morphology.nuclei_type_3,
                "miscellaneous": morphology.nuclei_type_4,
            },
            "mean_area_px2": round(morphology.nuclei_mean_area_px2, 2),
            "mean_perimeter_px": round(morphology.nuclei_mean_perimeter_px, 2),
            "mean_eccentricity": round(morphology.nuclei_mean_eccentricity, 3),
            "mean_circularity": round(morphology.nuclei_mean_circularity, 3),
            "interpretation": model_agreement.nuclear_interpretation,
        }

        # Format Gland Evidence
        gland_evidence = {
            "total_count": morphology.glands_total,
            "mean_area_pixels": round(morphology.glands_mean_area_px2, 2),
            "mean_perimeter_pixels": round(morphology.glands_mean_perimeter_px, 2),
            "mean_width_pixels": round(morphology.glands_mean_width_px, 2),
            "mean_height_pixels": round(morphology.glands_mean_height_px, 2),
            "mean_aspect_ratio": round(morphology.glands_mean_aspect_ratio, 3),
            "mean_circularity": round(morphology.glands_mean_circularity, 3),
            "interpretation": model_agreement.gland_interpretation,
        }

        # Format Reference Comparison
        reference_data = {
            "label": reference_result.label,
            "top_category": reference_result.top_category,
            "top_similarity_percent": reference_result.top_similarity_percent,
            "top_reference_id": reference_result.top_match_id,
            "insight": reference_result.clinical_insight,
            "comparisons": [c.model_dump() for c in reference_result.comparisons[:3]],
        }

        case_result = {
            "case_id": case_id,
            "timestamp": now_iso,
            "status": "completed",
            "image_quality": image_quality,
            "digepath": {
                "model_name": digepath_meta.get("model_name", "Digepath"),
                "architecture": digepath_meta.get("architecture", "ViT-L/16"),
                "embedding_dimension": digepath_meta.get("embedding_dimension", 1024),
                "device": digepath_meta.get("device", "cuda"),
                "status": "active",
            },
            "prediction": {
                "class": prediction_result.get("prediction", "UNKNOWN"),
                "confidence": round(prediction_result.get("confidence", 0.0), 4),
                "calibrated_confidence": round(uncertainty.calibrated_confidence, 4),
                "tumor_probability": round(prediction_result.get("tumor_probability", 0.0), 4),
                "binary_class": prediction_result.get("binary_prediction", "NON-TUM"),
                "multiclass_probabilities": prediction_result.get("multiclass_probabilities", {}),
            },
            "uncertainty": {
                "score": round(uncertainty.uncertainty_score, 4),
                "level": uncertainty.uncertainty_level,
                "entropy": round(uncertainty.entropy, 4),
                "normalized_entropy": round(uncertainty.normalized_entropy, 4),
                "ood_score": round(getattr(uncertainty, "ood_score", 0.0), 4),
                "ood_status": getattr(uncertainty, "ood_status", "IN_DISTRIBUTION"),
                "is_ood": getattr(uncertainty, "is_ood", False),
                "review_required": uncertainty.review_required,
                "message": uncertainty.abstention_message,
            },
            "model_agreement": {
                "level": model_agreement.level,
                "score": round(model_agreement.score, 4),
                "concordant_sources": model_agreement.concordant_sources,
                "discordant_sources": model_agreement.discordant_sources,
                "summary": model_agreement.summary,
            },
            "nuclear_evidence": nuclear_evidence,
            "gland_evidence": gland_evidence,
            "reference_comparison": reference_data,
            "priority_regions": [r.model_dump() for r in priority_regions],
            "visualizations": visualizations or {},
            "limitations": [
                "Research prototype for decision support; not an autonomous diagnostic device.",
                "Pathologist review recommended for all clinical correlations and staging.",
                "Visual and morphological features are AI-derived computational estimates.",
            ],
        }

        return case_result

    @classmethod
    def build_evidence_json(cls, case_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts purely factual verifiable metrics into evidence.json.
        """
        return {
            "case_id": case_result["case_id"],
            "timestamp": case_result["timestamp"],
            "prediction_class": case_result["prediction"]["class"],
            "prediction_confidence": case_result["prediction"]["confidence"],
            "calibrated_confidence": case_result["prediction"]["calibrated_confidence"],
            "tumor_probability": case_result["prediction"]["tumor_probability"],
            "uncertainty_score": case_result["uncertainty"]["score"],
            "uncertainty_level": case_result["uncertainty"]["level"],
            "ood_score": case_result["uncertainty"].get("ood_score", 0.0),
            "ood_status": case_result["uncertainty"].get("ood_status", "IN_DISTRIBUTION"),
            "agreement_level": case_result["model_agreement"]["level"],
            "nuclear_total_count": case_result["nuclear_evidence"]["total_count"],
            "nuclear_mean_area_px2": case_result["nuclear_evidence"]["mean_area_px2"],
            "gland_total_count": case_result["gland_evidence"]["total_count"],
            "gland_mean_circularity": case_result["gland_evidence"]["mean_circularity"],
            "reference_top_category": case_result["reference_comparison"]["top_category"],
            "reference_top_similarity_percent": case_result["reference_comparison"]["top_similarity_percent"],
            "priority_regions_count": len(case_result["priority_regions"]),
            "region_ids": [r["region_id"] for r in case_result["priority_regions"]],
        }
