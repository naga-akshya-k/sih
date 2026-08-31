"""
AI-Prioritized Region Ranking Engine.
Computes transparent priority scores to assist pathologists in triage and navigation.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field


class PriorityRanker:
    """
    Transparent scoring formula for AI-prioritized region ranking.
    Formula:
        priority_score = w_tum * P(tum) + w_nuc * S(nuclear_atypia) + w_unc * S(uncertainty) + w_gland * S(gland_distortion)
    """

    def __init__(
        self,
        w_tumor: float = 0.45,
        w_nuclear: float = 0.30,
        w_uncertainty: float = 0.15,
        w_gland: float = 0.10,
        high_threshold: float = 0.60,
        medium_threshold: float = 0.35,
    ):
        self.w_tumor = w_tumor
        self.w_nuclear = w_nuclear
        self.w_uncertainty = w_uncertainty
        self.w_gland = w_gland
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def calculate_priority(
        self,
        tumor_probability: float,
        uncertainty_score: float,
        nuclear_atypia_score: float = 0.0,
        gland_distortion_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates region priority score and qualitative tier ("HIGH", "MEDIUM", "LOW").
        """
        raw_score = (
            self.w_tumor * float(tumor_probability)
            + self.w_nuclear * float(nuclear_atypia_score)
            + self.w_uncertainty * float(uncertainty_score)
            + self.w_gland * float(gland_distortion_score)
        )
        score = float(max(0.0, min(1.0, raw_score)))

        if score >= self.high_threshold:
            tier = "HIGH"
            rationale = "High AI priority due to prominent tumor probability and nuclear atypia."
        elif score >= self.medium_threshold:
            tier = "MEDIUM"
            rationale = "Moderate AI priority due to intermediate cellular density and model uncertainty."
        else:
            tier = "LOW"
            rationale = "Low AI priority with regular morphological patterns."

        return {
            "priority_score": score,
            "priority_level": tier,
            "label": "AI-prioritized region",
            "rationale": rationale,
            "components": {
                "tumor_weight_contrib": round(self.w_tumor * tumor_probability, 4),
                "nuclear_weight_contrib": round(self.w_nuclear * nuclear_atypia_score, 4),
                "uncertainty_weight_contrib": round(self.w_uncertainty * uncertainty_score, 4),
                "gland_weight_contrib": round(self.w_gland * gland_distortion_score, 4),
            },
        }
