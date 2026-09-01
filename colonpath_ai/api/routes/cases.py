"""
Case Retrieval and Image Serving API Routes.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from api.schemas import CaseResultResponse, CaseSummaryItem
from api.services.case_service import CaseService

router = APIRouter(prefix="/cases", tags=["Cases"])
case_service = CaseService()


@router.get("", response_model=List[CaseSummaryItem])
def list_cases(limit: int = 50):
    cases = case_service.list_cases(limit=limit)
    return cases


@router.get("/{case_id}", response_model=Dict[str, Any])
def get_case_meta(case_id: str):
    meta = case_service.get_case_meta(case_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return meta


@router.get("/{case_id}/result", response_model=CaseResultResponse)
def get_case_result(case_id: str):
    result = case_service.get_case_result(case_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Result for case '{case_id}' not found.")
    return result


@router.get("/{case_id}/image")
def get_case_image(case_id: str):
    meta = case_service.get_case_meta(case_id)
    if not meta or not meta.get("image_path"):
        raise HTTPException(status_code=404, detail=f"Image for case '{case_id}' not found.")
    img_path = Path(meta["image_path"])
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file does not exist on disk.")
    return FileResponse(img_path)


@router.get("/{case_id}/visualization/{vis_type}")
def get_case_visualization(case_id: str, vis_type: str):
    result = case_service.get_case_result(case_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    visualizations = result.get("visualizations", {})
    if vis_type not in visualizations:
        valid_types = list(visualizations.keys()) or ["original", "glands", "nuclei", "regions", "uncertainty", "top_regions", "pseudo_3d"]
        raise HTTPException(status_code=404, detail=f"Visualization '{vis_type}' not found. Available: {valid_types}")

    file_path = Path(visualizations[vis_type])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Visualization file on disk not found: {file_path}")
    return FileResponse(file_path, media_type="image/png")


@router.get("/{case_id}/evidence", response_model=Dict[str, Any])
def get_case_evidence(case_id: str):
    """
    Returns the deterministic computational evidence payload (evidence.json) for a case.
    """
    result = case_service.get_case_result(case_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    evidence = {
        "case_id": result.get("case_id"),
        "timestamp": result.get("timestamp"),
        "prediction_class": result.get("prediction", {}).get("class"),
        "prediction_confidence": result.get("prediction", {}).get("confidence"),
        "calibrated_confidence": result.get("prediction", {}).get("calibrated_confidence"),
        "tumor_probability": result.get("prediction", {}).get("tumor_probability"),
        "uncertainty_score": result.get("uncertainty", {}).get("score"),
        "uncertainty_level": result.get("uncertainty", {}).get("level"),
        "ood_score": result.get("uncertainty", {}).get("ood_score", 0.0),
        "ood_status": result.get("uncertainty", {}).get("ood_status", "IN_DISTRIBUTION"),
        "agreement_level": result.get("model_agreement", {}).get("level"),
        "nuclear_total_count": result.get("nuclear_evidence", {}).get("total_count"),
        "nuclear_mean_area_px2": result.get("nuclear_evidence", {}).get("mean_area_px2"),
        "gland_total_count": result.get("gland_evidence", {}).get("total_count"),
        "gland_mean_circularity": result.get("gland_evidence", {}).get("mean_circularity"),
        "reference_top_category": result.get("reference_comparison", {}).get("top_category"),
        "reference_top_similarity_percent": result.get("reference_comparison", {}).get("top_similarity_percent"),
        "priority_regions_count": len(result.get("priority_regions", [])),
    }
    return evidence


@router.get("/{case_id}/report", response_model=Dict[str, Any])
def get_case_report(case_id: str):
    """
    Returns the structured MedGemma medical explanation report for a case.
    """
    result = case_service.get_case_result(case_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    explanation = result.get("explanation", {})
    return {
        "case_id": case_id,
        "explanation": explanation,
        "limitations": result.get("limitations", []),
        "status": result.get("status", "completed"),
    }

