"""
Region Navigation Engine for Mobile & Pathologist Viewers.
"""

from typing import List, Optional, Dict, Any
from .region_analyzer import RegionItem


class RegionNavigator:
    """
    Stateful and query-based navigation across AI-prioritized regions.
    """

    def __init__(self, regions: List[RegionItem]):
        # Keep regions strictly sorted by priority score descending
        self.regions = sorted(regions, key=lambda r: r.priority_score, reverse=True)
        self._id_map = {r.region_id: r for r in self.regions}

    @property
    def total_regions(self) -> int:
        return len(self.regions)

    def get_top(self, k: int = 3) -> List[RegionItem]:
        return self.regions[:k]

    def get_region(self, region_id: str) -> Optional[RegionItem]:
        return self._id_map.get(region_id)

    def get_next(self, current_region_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves the next priority region in sequence for mobile viewport navigation.
        """
        if not self.regions:
            return None

        if current_region_id is None:
            next_item = self.regions[0]
            next_pos = 0
        else:
            current_idx = -1
            for idx, r in enumerate(self.regions):
                if r.region_id == current_region_id:
                    current_idx = idx
                    break

            if current_idx == -1 or current_idx + 1 >= len(self.regions):
                next_pos = 0  # Loop back or keep at start
                next_item = self.regions[0]
            else:
                next_pos = current_idx + 1
                next_item = self.regions[next_pos]

        return {
            "region": next_item.model_dump(),
            "navigation": {
                "current_index": next_pos + 1,
                "total_regions": len(self.regions),
                "has_more": (next_pos + 1 < len(self.regions)),
                "coordinates": {
                    "x": next_item.x,
                    "y": next_item.y,
                    "width": next_item.width,
                    "height": next_item.height,
                    "center_x": next_item.x + next_item.width // 2,
                    "center_y": next_item.y + next_item.height // 2,
                },
            },
        }

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [r.model_dump() for r in self.regions]
