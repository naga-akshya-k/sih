"""
Analysis API Route for Full H&E Histopathology Evaluation.
"""

import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from api.schemas import CaseResultResponse
from api.services.case_service import CaseService

router = APIRouter(tags=["Analysis"])
case_service = CaseService()

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "outputs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze", response_model=CaseResultResponse, status_code=status.HTTP_200_OK)
async def analyze_image(
    image: UploadFile = File(..., description="H&E Histopathology image (PNG, JPG, BMP, TIF)"),
    case_id: Optional[str] = Form(None, description="Optional Case Identifier"),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    # Save uploaded file temporarily
    file_ext = Path(image.filename).suffix or ".png"
    safe_cid = case_id or Path(image.filename).stem
    save_path = UPLOAD_DIR / f"{safe_cid}{file_ext}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        result = case_service.analyze_image(image_path=save_path, case_id=safe_cid)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")
