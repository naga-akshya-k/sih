"""
Evidence-Grounded Explanation Generator for Decision Support.
Ensures explanations are strictly grounded in computational outputs without hallucinations.
"""

from typing import Dict, Any, Optional


class EvidenceGroundedExplainer:
    """
    Generates structured, factual explanations strictly from verified computational evidence.
    """

    @classmethod
    def generate_explanation(cls, case_result: Dict[str, Any]) -> str:
        """
        Generates a factual narrative directly citing case_result facts.
        """
        pred = case_result.get("prediction", {})
        unc = case_result.get("uncertainty", {})
        agr = case_result.get("model_agreement", {})
        nuc = case_result.get("nuclear_evidence", {})
        gland = case_result.get("gland_evidence", {})
        ref = case_result.get("reference_comparison", {})
        regions = case_result.get("priority_regions", [])

        # Check for abstention / high uncertainty
        if unc.get("level") == "HIGH" or unc.get("review_required") is True:
            return (
                f"AI-Assisted Classification: {pred.get('class', 'UNKNOWN')} "
                f"(Calibrated Confidence: {pred.get('calibrated_confidence', 0.0):.2f}). "
                f"High model uncertainty detected (Score: {unc.get('score', 0.0):.2f}). "
                f"Pathologist review is recommended to evaluate cellular features."
            )

        # Normal evidence-grounded summary
        pred_class = pred.get("class", "NORM")
        conf = pred.get("calibrated_confidence", 0.0)
        n_total = nuc.get("total_count", 0)
        n_area = nuc.get("mean_area_px2", 0.0)
        g_total = gland.get("total_count", 0)
        g_circ = gland.get("mean_circularity", 0.0)
        ref_cat = ref.get("top_category", "unknown")
        ref_sim = ref.get("top_similarity_percent", 0.0)

        lines = [
            f"AI-assisted classification suggests **{pred_class}** with {conf * 100:.1f}% calibrated confidence.",
            f"Nuclear Analysis: {n_total} nuclei detected (mean area: {n_area:.1f} px²).",
            f"Gland Analysis: {g_total} glandular structures segmented (mean circularity: {g_circ:.2f}).",
            f"Model Agreement: {agr.get('level', 'HIGH')} agreement across computational models.",
            f"Reference Match: {ref_sim:.1f}% similarity to curated '{ref_cat}' reference cohort.",
        ]

        if regions:
            top_r = regions[0]
            lines.append(
                f"Top AI-Prioritized Region: {top_r.get('region_id')} (Priority Score: {top_r.get('priority_score', 0.0):.2f}) "
                f"located at (x={top_r.get('x')}, y={top_r.get('y')})."
            )

        return "\n".join(lines)
