"""
Pathologist-in-the-Loop Review and Notes Routes.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from api.schemas import ReviewRequest, NoteRequest
from api.services.case_service import CaseService

router = APIRouter(prefix="/cases/{case_id}", tags=["Pathologist Review"])
case_service = CaseService()


@router.post("/review")
def submit_review(case_id: str, payload: ReviewRequest):
    meta = case_service.get_case_meta(case_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    valid_actions = ["MARK_REVIEWED", "FLAG_REGION", "ADD_NOTE", "REQUEST_REANALYSIS"]
    if payload.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{payload.action}'. Allowed: {valid_actions}",
        )

    case_service.add_review(
        case_id=case_id,
        action=payload.action,
        notes=payload.notes or "",
        pathologist_id=payload.pathologist_id or "Dr. Pathologist",
    )
    return {"status": "success", "case_id": case_id, "action": payload.action}


@router.post("/notes")
def add_case_note(case_id: str, payload: NoteRequest):
    meta = case_service.get_case_meta(case_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    case_service.add_note(
        case_id=case_id,
        note_text=payload.note_text,
        author=payload.author or "Pathologist",
    )
    return {"status": "success", "case_id": case_id}


@router.get("/notes")
def list_case_notes(case_id: str):
    meta = case_service.get_case_meta(case_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return case_service.get_notes(case_id)


@router.post("/feedback")
def submit_feedback(case_id: str, payload: Dict[str, Any]):
    """
    Records pathologist validation feedback (CORRECT, INCORRECT, UNCERTAIN, REVIEW_REQUIRED).
    """
    meta = case_service.get_case_meta(case_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    feedback_label = payload.get("feedback", "REVIEW_REQUIRED")
    notes = payload.get("notes", "")
    author = payload.get("pathologist_id", "Dr. Pathologist")

    case_service.add_review(
        case_id=case_id,
        action=f"FEEDBACK_{feedback_label}",
        notes=notes,
        pathologist_id=author,
    )
    return {
        "status": "success",
        "case_id": case_id,
        "feedback": feedback_label,
        "recorded": True,
    }

