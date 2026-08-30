"""
Unit tests for anti-hallucination evidence validator.
"""

from agent.evidence_validator import EvidenceValidator


def test_evidence_validator_valid():
    case_res = {
        "prediction": {"class": "TUM", "confidence": 0.92, "calibrated_confidence": 0.90},
        "nuclear_evidence": {"total_count": 117},
        "gland_evidence": {"total_count": 2},
        "priority_regions": [{"region_id": "R_01"}, {"region_id": "R_02"}],
    }

    valid_text = (
        "AI-assisted classification suggests TUM with high confidence. "
        "Nuclear Analysis: 117 nuclei detected. "
        "Gland Analysis: 2 glands segmented. "
        "Top region is R_01."
    )
    res = EvidenceValidator.validate(valid_text, case_res)
    assert res.is_valid is True
    assert len(res.errors) == 0


def test_evidence_validator_reject_hallucination():
    case_res = {
        "prediction": {"class": "TUM", "confidence": 0.92, "calibrated_confidence": 0.90},
        "nuclear_evidence": {"total_count": 117},
        "gland_evidence": {"total_count": 2},
        "priority_regions": [{"region_id": "R_01"}, {"region_id": "R_02"}],
    }

    hallucinated_text = (
        "This is confirmed cancer with 100% accurate diagnosis. "
        "We counted 500 nuclei in region R_99."
    )
    res = EvidenceValidator.validate(hallucinated_text, case_res)
    assert res.is_valid is False
    assert len(res.errors) >= 3
