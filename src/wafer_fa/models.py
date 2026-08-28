from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WaferGeometry:
    center_x: float
    center_y: float
    radius: float
    detection_method: str


@dataclass
class FeatureBundle:
    particle_count: int
    radial: np.ndarray
    angular: np.ndarray
    density: np.ndarray
    summary: np.ndarray
    cluster: np.ndarray
    normalized_points: np.ndarray
    geometry: WaferGeometry | None = None


@dataclass
class CaseRecord:
    id: int
    image_path: Path
    normalized_path: Path | None
    comment: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class SearchResult:
    case: CaseRecord
    score: float
    components: dict[str, float]
