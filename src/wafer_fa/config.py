from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FeatureConfig:
    normalized_size: int = 512
    wafer_radius_ratio: float = 0.46
    radial_bins: int = 20
    angular_bins: int = 36
    density_grid: int = 16
    red_hsv_low_1: tuple[int, int, int] = (0, 90, 80)
    red_hsv_high_1: tuple[int, int, int] = (12, 255, 255)
    red_hsv_low_2: tuple[int, int, int] = (168, 90, 80)
    red_hsv_high_2: tuple[int, int, int] = (179, 255, 255)
    min_particle_area_px: int = 1
    max_particle_area_ratio: float = 0.0025
    dbscan_eps_norm: float = 0.075
    dbscan_min_samples: int = 4
    dbscan_max_points: int = 1800


@dataclass(frozen=True)
class SimilarityWeights:
    density: float = 0.35
    radial: float = 0.20
    angular: float = 0.10
    summary: float = 0.15
    cluster: float = 0.10
    particle_count: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "density": self.density,
            "radial": self.radial,
            "angular": self.angular,
            "summary": self.summary,
            "cluster": self.cluster,
            "particle_count": self.particle_count,
        }


@dataclass(frozen=True)
class AppConfig:
    root: Path = field(default_factory=lambda: Path.cwd())
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    weights: SimilarityWeights = field(default_factory=SimilarityWeights)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "wafer_fa.db"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"
