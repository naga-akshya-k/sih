"""
Region Service Layer for AI-Prioritized Region Navigation.
"""

from typing import Optional, List, Dict, Any
from storage.case_repository import CaseRepository
from regions.region_analyzer import RegionItem
from regions.region_navigator import RegionNavigator


class RegionService:
    def __init__(self, repository: Optional[CaseRepository] = None):
        self.repository = repository or CaseRepository()

    def get_regions(self, case_id: str) -> List[Dict[str, Any]]:
        result = self.repository.get_case_result(case_id)
        if not result or "priority_regions" not in result:
            return []
        return result["priority_regions"]

    def get_region(self, case_id: str, region_id: str) -> Optional[Dict[str, Any]]:
        regions = self.get_regions(case_id)
        for r in regions:
            if r.get("region_id") == region_id:
                return r
        return None

    def get_next_region(self, case_id: str, current_region_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raw_regions = self.get_regions(case_id)
        if not raw_regions:
            return None
        region_items = [RegionItem(**r) for r in raw_regions]
        navigator = RegionNavigator(region_items)
        return navigator.get_next(current_region_id)
