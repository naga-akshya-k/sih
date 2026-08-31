"""
Pathologist Copilot & MedGemma VLM API Routes.
Provides interactive evidence-grounded Q&A for pathologists.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services.case_service import CaseService
from agent.medgemma_vlm import MedGemmaVLM

router = APIRouter(prefix="/copilot", tags=["Pathologist Copilot & MedGemma"])

case_service = CaseService()
medgemma_vlm = MedGemmaVLM()


class CopilotQuestionRequest(BaseModel):
    case_id: str = Field(..., description="Target Case ID (e.g. CASE_DEMO_00000)")
    question: str = Field(..., description="Pathologist clinical inquiry")
    region_id: Optional[str] = Field(None, description="Optional focused region ID (e.g. R_01)")


class CopilotAnswerResponse(BaseModel):
    case_id: str
    question: str
    selected_region_id: Optional[str] = None
    answer: str
    model: str
    validated: bool
    validation_errors: list = []


@router.post("/ask", response_model=CopilotAnswerResponse)
def ask_copilot(req: CopilotQuestionRequest):
    """
    Queries Google MedGemma 1.5 4B IT / Pathologist Copilot regarding computational case evidence.
    """
    case_result = case_service.get_case_result(req.case_id)
    if not case_result:
        raise HTTPException(
            status_code=404,
            detail=f"Case '{req.case_id}' not found. Run analysis first via POST /analyze.",
        )

    ans = medgemma_vlm.answer_copilot_question(
        question=req.question,
        case_result=case_result,
        selected_region_id=req.region_id,
    )

    return ans
