"""
AI-Prioritized Region Analysis and Navigation Routes.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from api.schemas import RegionDetailSchema, NextRegionResponse
from api.services.region_service import RegionService

router = APIRouter(prefix="/cases/{case_id}/regions", tags=["Regions"])
region_service = RegionService()


@router.get("", response_model=List[RegionDetailSchema])
def get_all_regions(case_id: str):
    regions = region_service.get_regions(case_id)
    if not regions:
        raise HTTPException(status_code=404, detail=f"No regions found for case '{case_id}'.")
    return regions


@router.get("/next", response_model=NextRegionResponse)
def get_next_region(case_id: str, current_region_id: Optional[str] = Query(None)):
    nav_data = region_service.get_next_region(case_id, current_region_id)
    if not nav_data:
        raise HTTPException(status_code=404, detail=f"No navigation data available for case '{case_id}'.")
    return NextRegionResponse(case_id=case_id, **nav_data)


@router.get("/{region_id}", response_model=RegionDetailSchema)
def get_region_detail(case_id: str, region_id: str):
    region = region_service.get_region(case_id, region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found in case '{case_id}'.")
    return region
