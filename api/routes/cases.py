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
