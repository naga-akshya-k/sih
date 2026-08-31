"""
Comprehensive Unit Tests for MedGemma Pathologist Copilot Q&A.
Verifies that the Copilot accurately answers inquiries across all clinical domains.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
CASE_ID = "CASE_DEMO_00000"


def test_copilot_all_pathologist_questions():
    test_questions = [
        ("Why was region R_03 prioritized?", "Priority Score"),
        ("What is the AI prediction and tumor probability?", "calibrated confidence"),
        ("What nuclear abnormalities and cell types were detected?", "Nuclear Cytopathology"),
        ("What gland features were segmented by U-Net?", "Glandular Histomorphometry"),
        ("Why is the model uncertain and what is the entropy score?", "Model Reliability"),
        ("What is the model agreement and are there conflicts?", "Multi-Source Consensus"),
        ("Which reference case is most similar?", "Reference Retrieval"),
        ("Which region should I review next?", "Next Region Recommendation"),
        ("What is the image quality and blur variance?", "Image Quality Assessment"),
        ("What clinical recommendations and limitations apply?", "Clinical Recommendations"),
        ("Give me an executive summary of this case.", "Case Summary"),
    ]

    for q_text, expected_keyword in test_questions:
        res = client.post(
            "/copilot/ask",
            json={
                "case_id": CASE_ID,
                "question": q_text,
            },
        )
        assert res.status_code == 200, f"Failed on question: {q_text}"
        data = res.json()
        assert data["case_id"] == CASE_ID
        assert data["validated"] is True, f"Anti-hallucination failed on: {q_text}"
        assert expected_keyword.lower() in data["answer"].lower(), f"Missing keyword '{expected_keyword}' in answer: {data['answer']}"
