"""
SQLite Storage Engine for Cases, Regions, Reviews, and Pathologist Notes.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Union

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "colonpath.db"


class Database:
    """
    Lightweight SQLite database manager.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Cases table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prediction_class TEXT,
                    confidence REAL,
                    calibrated_confidence REAL,
                    tumor_probability REAL,
                    uncertainty_score REAL,
                    uncertainty_level TEXT,
                    agreement_level TEXT,
                    review_status TEXT DEFAULT 'PENDING',
                    image_path TEXT,
                    result_json_path TEXT,
                    evidence_json_path TEXT
                )
            """)

            # Reviews & Pathologist Actions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    notes TEXT,
                    pathologist_id TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases(case_id)
                )
            """)

            # Pathologist Notes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    note_text TEXT NOT NULL,
                    author TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases(case_id)
                )
            """)
            conn.commit()
