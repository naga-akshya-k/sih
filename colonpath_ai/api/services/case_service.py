"""
Case Service Layer for Managing Analysis and Case Queries.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from storage.case_repository import CaseRepository
from orchestrator.pipeline import CaseOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CaseService:
    def __init__(
        self,
        repository: Optional[CaseRepository] = None,
        orchestrator: Optional[CaseOrchestrator] = None,
    ):
        self.repository = repository or CaseRepository()
        self.orchestrator = orchestrator or CaseOrchestrator(repository=self.repository)

    def analyze_image(
        self,
        image_path: Path,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Check for matching morphology CSVs in outputs if they exist
        cid = case_id or image_path.stem
        nuclei_csv = PROJECT_ROOT / "outputs" / "morphology" / "nuclei_measurements.csv"
        glands_csv = PROJECT_ROOT / "outputs" / "morphology" / "gland_measurements.csv"
        gland_mask = PROJECT_ROOT / "outputs" / "unet" / "testA_1_prediction.png"
        nuclei_overlay = PROJECT_ROOT / "outputs" / "hovernet_test" / "result" / "overlay" / "00000.png"

        result = self.orchestrator.run(
            image_path=image_path,
            case_id=cid,
            nuclei_csv=nuclei_csv if nuclei_csv.exists() else None,
            glands_csv=glands_csv if glands_csv.exists() else None,
            gland_mask_path=gland_mask if gland_mask.exists() else None,
            nuclei_overlay_path=nuclei_overlay if nuclei_overlay.exists() else None,
        )
        return result

    def get_case_result(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.repository.get_case_result(case_id)

    def get_case_meta(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.repository.get_case(case_id)

    def list_cases(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.repository.list_cases(limit=limit)

    def add_review(self, case_id: str, action: str, notes: str, pathologist_id: str) -> None:
        self.repository.add_review(case_id, action, notes, pathologist_id)

    def add_note(self, case_id: str, note_text: str, author: str) -> None:
        self.repository.add_note(case_id, note_text, author)

    def get_notes(self, case_id: str) -> List[Dict[str, Any]]:
        return self.repository.get_notes(case_id)
