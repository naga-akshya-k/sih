"""
End-to-End Pipeline Integration Test for COLONPATH-AI.
Runs the complete sequence on test case 00000.png and verifies all deliverables.
"""

from pathlib import Path
import json
import pytest
from orchestrator.pipeline import CaseOrchestrator

TEST_IMAGE = Path(__file__).resolve().parents[1] / "outputs" / "hovernet_test" / "input" / "00000.png"
NUCLEI_CSV = Path(__file__).resolve().parents[1] / "outputs" / "morphology" / "nuclei_measurements.csv"
GLANDS_CSV = Path(__file__).resolve().parents[1] / "outputs" / "morphology" / "gland_measurements.csv"
GLAND_MASK = Path(__file__).resolve().parents[1] / "outputs" / "unet" / "testA_1_prediction.png"
NUCLEI_OVERLAY = Path(__file__).resolve().parents[1] / "outputs" / "hovernet_test" / "result" / "overlay" / "00000.png"


def test_end_to_end_pipeline():
    if not TEST_IMAGE.exists():
        pytest.skip("Test image 00000.png not found")

    orchestrator = CaseOrchestrator()
    case_id = "E2E_VERIFICATION_001"

    result = orchestrator.run(
        image_path=TEST_IMAGE,
        case_id=case_id,
        nuclei_csv=NUCLEI_CSV if NUCLEI_CSV.exists() else None,
        glands_csv=GLANDS_CSV if GLANDS_CSV.exists() else None,
        gland_mask_path=GLAND_MASK if GLAND_MASK.exists() else None,
        nuclei_overlay_path=NUCLEI_OVERLAY if NUCLEI_OVERLAY.exists() else None,
    )

    # 1. Verify schema keys
    required_keys = [
        "case_id",
        "image_quality",
        "digepath",
        "prediction",
        "uncertainty",
        "model_agreement",
        "nuclear_evidence",
        "gland_evidence",
        "reference_comparison",
        "priority_regions",
        "visualizations",
        "limitations",
    ]
    for k in required_keys:
        assert k in result, f"Missing required key: {k}"

    # 2. Verify disk artifacts
    case_dir = Path(__file__).resolve().parents[1] / "outputs" / "cases" / case_id
    assert (case_dir / "case_result.json").exists()
    assert (case_dir / "evidence.json").exists()

    # 3. Verify all 7 genuine visualizations were rendered
    vis = result["visualizations"]
    for vis_type in ["original", "glands", "nuclei", "regions", "uncertainty", "top_regions", "pseudo_3d"]:
        assert vis_type in vis
        assert Path(vis[vis_type]).exists()

    # 4. Verify evidence explanation validation
    assert "explanation" in result
    assert result["explanation"]["validated"] is True
