from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .config import AppConfig
from .db import CaseDatabase
from .features import extract_features
from .image_io import imread_unicode, imwrite_unicode, normalize_wafer
from .models import FeatureBundle, SearchResult
from .similarity import compare_features

FEATURE_VERSION = "spatial-v1"


class WaferFAService:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig()
        self.db = CaseDatabase(self.cfg.db_path)
        self.ensure_storage()

    def ensure_storage(self) -> None:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.image_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.db.init()

    def analyze(self, image_path: str | Path) -> tuple[FeatureBundle, Any]:
        image = imread_unicode(image_path)
        normalized = normalize_wafer(image, self.cfg.feature)
        features = extract_features(normalized, self.cfg.feature)
        return features, normalized

    def add_case(
        self,
        image_path: str | Path,
        comment: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        source = Path(image_path)
        if not source.exists():
            raise FileNotFoundError(source)
        metadata = metadata or {}
        features, normalized = self.analyze(source)
        if features.particle_count == 0:
            raise ValueError("No red particles detected inside wafer region")

        suffix = source.suffix.lower() if source.suffix else ".png"
        case_token = uuid.uuid4().hex[:12]
        stored = self.cfg.image_dir / f"{case_token}{suffix}"
        shutil.copy2(source, stored)
        normalized_path = self.cfg.normalized_dir / f"{case_token}.png"
        imwrite_unicode(normalized_path, normalized.image)
        return self.db.insert(
            stored.resolve(),
            normalized_path.resolve(),
            comment,
            metadata,
            features,
            FEATURE_VERSION,
        )

    def search(self, image_path: str | Path, top_k: int = 3) -> list[SearchResult]:
        query, _ = self.analyze(image_path)
        if query.particle_count == 0:
            raise ValueError("No red particles detected inside wafer region")
        results: list[SearchResult] = []
        for record, candidate in self.db.iter_cases_with_features():
            score, components = compare_features(query, candidate, self.cfg.weights)
            results.append(SearchResult(record, score, components))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: max(1, top_k)]

    def rebuild(self) -> dict[str, int]:
        ok = 0
        failed = 0
        for record, _ in list(self.db.iter_cases_with_features()):
            try:
                features, normalized = self.analyze(record.image_path)
                normalized_path = record.normalized_path or (
                    self.cfg.normalized_dir / f"case_{record.id}.png"
                )
                imwrite_unicode(normalized_path, normalized.image)
                self.db.update_features(record.id, normalized_path, features, FEATURE_VERSION)
                ok += 1
            except Exception:
                failed += 1
        return {"rebuilt": ok, "failed": failed}
