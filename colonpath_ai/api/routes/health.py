"""
Health Check API Route.
"""

from fastapi import APIRouter
import torch
from api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="healthy",
        service="COLONPATH-AI Multimodal Backend",
        version="1.0.0",
        device=device,
        models_ready=True,
    )
