from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from .codec import pack_features, unpack_features
from .models import CaseRecord, FeatureBundle


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    normalized_path TEXT,
    comment TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    feature_blob BLOB NOT NULL,
    feature_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at);
"""


class CaseDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def init(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)

    def insert(
        self,
        image_path: Path,
        normalized_path: Path | None,
        comment: str,
        metadata: dict,
        features: FeatureBundle,
        feature_version: str,
    ) -> int:
        self.init()
        with self.connect() as con:
            cur = con.execute(
                """
                INSERT INTO cases(image_path, normalized_path, comment, metadata_json,
                                  feature_blob, feature_version)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    str(image_path),
                    str(normalized_path) if normalized_path else None,
                    comment,
                    json.dumps(metadata, ensure_ascii=False),
                    pack_features(features),
                    feature_version,
                ),
            )
            return int(cur.lastrowid)

    def update_features(
        self,
        case_id: int,
        normalized_path: Path | None,
        features: FeatureBundle,
        feature_version: str,
    ) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE cases SET normalized_path=?, feature_blob=?, feature_version=? WHERE id=?",
                (
                    str(normalized_path) if normalized_path else None,
                    pack_features(features),
                    feature_version,
                    case_id,
                ),
            )

    def iter_cases_with_features(self) -> Iterator[tuple[CaseRecord, FeatureBundle]]:
        self.init()
        with self.connect() as con:
            rows = con.execute("SELECT * FROM cases ORDER BY id ASC").fetchall()
        for row in rows:
            meta = json.loads(row["metadata_json"] or "{}")
            record = CaseRecord(
                id=int(row["id"]),
                image_path=Path(row["image_path"]),
                normalized_path=Path(row["normalized_path"]) if row["normalized_path"] else None,
                comment=row["comment"] or "",
                metadata=meta,
                created_at=row["created_at"] or "",
            )
            yield record, unpack_features(row["feature_blob"])

    def count(self) -> int:
        self.init()
        with self.connect() as con:
            row = con.execute("SELECT COUNT(*) AS n FROM cases").fetchone()
        return int(row["n"])
