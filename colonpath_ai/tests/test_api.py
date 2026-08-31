"""
Unit and integration tests for FastAPI backend endpoints.
"""

from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
TEST_IMAGE = Path(__file__).resolve().parents[1] / "outputs" / "hovernet_test" / "input" / "00000.png"


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True


def test_analyze_and_case_lifecycle():
    if not TEST_IMAGE.exists():
        return

    with open(TEST_IMAGE, "rb") as f:
        r = client.post("/analyze", files={"image": ("00000.png", f, "image/png")}, data={"case_id": "UNIT_TEST_CASE"})
    assert r.status_code == 200
    res = r.json()
    assert res["case_id"] == "UNIT_TEST_CASE"
    assert "prediction" in res
    assert "uncertainty" in res
    assert "priority_regions" in res

    # Test GET result
    r_get = client.get("/cases/UNIT_TEST_CASE/result")
    assert r_get.status_code == 200

    # Test GET regions
    r_reg = client.get("/cases/UNIT_TEST_CASE/regions")
    assert r_reg.status_code == 200

    # Test Next region
    r_next = client.get("/cases/UNIT_TEST_CASE/regions/next")
    assert r_next.status_code == 200

    # Test review
    r_rev = client.post("/cases/UNIT_TEST_CASE/review", json={"action": "MARK_REVIEWED", "notes": "Test review"})
    assert r_rev.status_code == 200

    # Test Copilot & MedGemma endpoint
    r_copilot = client.post(
        "/copilot/ask",
        json={
            "case_id": "UNIT_TEST_CASE",
            "question": "Why was this region prioritized?",
        },
    )
    assert r_copilot.status_code == 200
    c_data = r_copilot.json()
    assert c_data["case_id"] == "UNIT_TEST_CASE"
    assert "answer" in c_data
    assert c_data["validated"] is True
