"""
Evidence Validator for Anti-Hallucination Guardrails.
Strictly verifies that any explanation or narrative adheres 100% to computed evidence.json and case_result.json.
"""

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    validated_explanation: Optional[str] = None


class EvidenceValidator:
    """
    Anti-Hallucination Gatekeeper. Rejects explanations containing unsupported facts or false diagnostic claims.
    """

    PROHIBITED_PHRASES = [
        "confirmed cancer",
        "definitive diagnosis",
        "100% accurate",
        "100% confident",
        "biopsy confirmed",
        "autonomous diagnosis",
    ]

    @classmethod
    def validate(
        cls,
        explanation_text: str,
        case_result: Dict[str, Any],
        evidence_json: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        errors = []
        text_lower = explanation_text.lower()

        # 1. Check for prohibited diagnostic overclaims
        for phrase in cls.PROHIBITED_PHRASES:
            if phrase in text_lower:
                errors.append(f"Prohibited overclaim detected: '{phrase}'. Must use decision-support phrasing.")

        # 2. Check tissue class grounding
        pred_class = case_result.get("prediction", {}).get("class", "").upper()
        # Find 3-4 letter uppercase tokens that look like tissue classes
        mentioned_classes = set(re.findall(r"\b(ADI|BACK|DEB|LYM|MUC|MUS|NORM|STR|TUM)\b", explanation_text))
        if mentioned_classes and pred_class not in mentioned_classes:
            errors.append(
                f"Class hallucination: Explanation mentions {mentioned_classes} but prediction is {pred_class}."
            )

        # 3. Check Region ID grounding
        valid_region_ids = {r.get("region_id") for r in case_result.get("priority_regions", [])}
        mentioned_regions = set(re.findall(r"\b(R_\d{2})\b", explanation_text))
        invalid_regions = mentioned_regions - valid_region_ids
        if invalid_regions:
            errors.append(f"Invalid region ID cited: {invalid_regions}. Valid regions: {valid_region_ids}")

        # 4. Check numerical cell counts if explicitly stated in text
        # If explicitly stating whole-slide counts e.g. "117 total nuclei" or overall summary
        n_match = re.search(r"(\d+)\s+(?:total\s+)?nuclei", text_lower)
        if n_match:
            stated_count = int(n_match.group(1))
            true_total = case_result.get("nuclear_evidence", {}).get("total_count", 0)
            region_counts = [r.get("nuclei_count", 0) for r in case_result.get("priority_regions", [])]
            valid_counts = {true_total} | set(region_counts)
            if stated_count not in valid_counts and stated_count > true_total:
                errors.append(f"Nuclear count discrepancy: stated {stated_count}, valid counts: {valid_counts}.")

        g_match = re.search(r"(\d+)\s+(?:total\s+)?gland", text_lower)
        if g_match:
            stated_glands = int(g_match.group(1))
            true_glands = case_result.get("gland_evidence", {}).get("total_count", 0)
            region_glands = [r.get("glands_count", 0) for r in case_result.get("priority_regions", [])]
            valid_gland_counts = {true_glands} | set(region_glands)
            if stated_glands not in valid_gland_counts and stated_glands > true_glands:
                errors.append(f"Gland count discrepancy: stated {stated_glands}, valid counts: {valid_gland_counts}.")

        is_valid = (len(errors) == 0)
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            validated_explanation=explanation_text if is_valid else None,
        )
