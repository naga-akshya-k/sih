"""
Case Repository for Database CRUD and Review Operations.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from .database import Database


class CaseRepository:
    """
    Data Access Object for Case lifecycle and pathologist-in-the-loop review.
    """

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def save_case(
        self,
        case_id: str,
        case_result: Dict[str, Any],
        result_json_path: Path,
        evidence_json_path: Path,
        image_path: Path,
    ) -> None:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cases (
                    case_id, timestamp, prediction_class, confidence,
                    calibrated_confidence, tumor_probability, uncertainty_score,
                    uncertainty_level, agreement_level, review_status,
                    image_path, result_json_path, evidence_json_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case_id,
                case_result.get("timestamp", datetime.now(timezone.utc).isoformat()),
                case_result.get("prediction", {}).get("class"),
                case_result.get("prediction", {}).get("confidence"),
                case_result.get("prediction", {}).get("calibrated_confidence"),
                case_result.get("prediction", {}).get("tumor_probability"),
                case_result.get("uncertainty", {}).get("score"),
                case_result.get("uncertainty", {}).get("level"),
                case_result.get("model_agreement", {}).get("level"),
                "PENDING",
                str(image_path),
                str(result_json_path),
                str(evidence_json_path),
            ))
            conn.commit()

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_case_result(self, case_id: str) -> Optional[Dict[str, Any]]:
        case_meta = self.get_case(case_id)
        if not case_meta or not case_meta.get("result_json_path"):
            return None
        path = Path(case_meta["result_json_path"])
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_cases(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def add_review(
        self,
        case_id: str,
        action: str,  # "MARK_REVIEWED", "FLAG_REGION", "ADD_NOTE", "REQUEST_REANALYSIS"
        notes: str = "",
        pathologist_id: str = "Dr. Pathologist",
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reviews (case_id, timestamp, action, notes, pathologist_id)
                VALUES (?, ?, ?, ?, ?)
            """, (case_id, now_iso, action, notes, pathologist_id))

            # Update case review_status
            new_status = "REVIEWED" if action == "MARK_REVIEWED" else "FLAGGED" if action == "FLAG_REGION" else "PENDING"
            cursor.execute("UPDATE cases SET review_status = ? WHERE case_id = ?", (new_status, case_id))
            conn.commit()

    def add_note(self, case_id: str, note_text: str, author: str = "Pathologist") -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes (case_id, timestamp, note_text, author)
                VALUES (?, ?, ?, ?)
            """, (case_id, now_iso, note_text, author))
            conn.commit()

    def get_notes(self, case_id: str) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM notes WHERE case_id = ? ORDER BY timestamp ASC", (case_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
