"""
Unit tests for AI-prioritized region ranking and Next Region navigation.
"""

from regions.region_analyzer import RegionItem
from regions.priority_ranking import PriorityRanker
from regions.region_navigator import RegionNavigator


def test_priority_ranker():
    ranker = PriorityRanker()
    high_res = ranker.calculate_priority(
        tumor_probability=0.95,
        uncertainty_score=0.10,
        nuclear_atypia_score=0.85,
    )
    assert high_res["priority_level"] == "HIGH"
    assert high_res["priority_score"] >= 0.60


def test_region_navigator():
    r1 = RegionItem(
        region_id="R_01",
        index=1,
        x=0,
        y=0,
        width=128,
        height=128,
        prediction="TUM",
        confidence=0.95,
        tumor_probability=0.95,
        uncertainty_score=0.1,
        uncertainty_level="LOW",
        priority_score=0.85,
        priority_level="HIGH",
        rationale="High tumor likelihood",
    )
    r2 = RegionItem(
        region_id="R_02",
        index=2,
        x=128,
        y=0,
        width=128,
        height=128,
        prediction="NORM",
        confidence=0.90,
        tumor_probability=0.05,
        uncertainty_score=0.1,
        uncertainty_level="LOW",
        priority_score=0.15,
        priority_level="LOW",
        rationale="Normal",
    )

    nav = RegionNavigator([r2, r1])  # should sort r1 first
    first = nav.get_next()
    assert first["region"]["region_id"] == "R_01"
    assert first["navigation"]["coordinates"]["center_x"] == 64

    second = nav.get_next("R_01")
    assert second["region"]["region_id"] == "R_02"
