from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .image_io import imwrite_unicode


PATTERNS = ("center", "line", "ring", "arc", "random")


def _clip_disk(points: np.ndarray, max_r: float = 0.92) -> np.ndarray:
    if len(points) == 0:
        return points
    return points[np.linalg.norm(points, axis=1) <= max_r]


def make_points(pattern: str, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if pattern == "center":
        pts = rng.normal(0.0, 0.16, size=(n, 2))
    elif pattern == "line":
        angle = rng.uniform(-0.25, 0.25) + math.pi / 2
        t = rng.uniform(-0.72, 0.72, size=n)
        normal = rng.normal(0.0, 0.018, size=n)
        direction = np.array([math.cos(angle), math.sin(angle)])
        perpendicular = np.array([-direction[1], direction[0]])
        offset = rng.normal(0.12, 0.035, size=2)
        pts = t[:, None] * direction + normal[:, None] * perpendicular + offset
    elif pattern == "ring":
        theta = rng.uniform(-math.pi, math.pi, size=n)
        radius = rng.normal(0.66, 0.035, size=n)
        pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    elif pattern == "arc":
        center_angle = rng.uniform(-math.pi, math.pi)
        theta = rng.normal(center_angle, 0.42, size=n)
        radius = rng.normal(0.74, 0.028, size=n)
        pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    elif pattern == "random":
        theta = rng.uniform(-math.pi, math.pi, size=n)
        radius = np.sqrt(rng.uniform(0.0, 0.84**2, size=n))
        pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return _clip_disk(pts.astype(np.float32))


def render_wafer(
    points: np.ndarray,
    size: int = 640,
    radius_ratio: float = 0.43,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    image = np.full((size, size, 3), background, dtype=np.uint8)
    center = np.array([size / 2.0, size / 2.0])
    radius = size * radius_ratio
    cv2.circle(image, tuple(center.astype(int)), int(radius), (20, 20, 20), 3)
    for x, y in points:
        px = int(round(center[0] + x * radius))
        py = int(round(center[1] + y * radius))
        cv2.circle(image, (px, py), max(2, size // 240), (0, 0, 255), -1, cv2.LINE_AA)
    return image


def generate_demo_set(
    out_dir: str | Path,
    count_per_pattern: int = 8,
    seed: int = 100,
) -> list[tuple[Path, str, dict[str, str]]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[Path, str, dict[str, str]]] = []
    comments = {
        "center": "Center concentration demo; verify center-zone contamination source.",
        "line": "Directional/line signature demo; check handling path, gas flow or scratch-like source.",
        "ring": "Ring distribution demo; check edge/radial process signature.",
        "arc": "Arc/sector distribution demo; check localized chamber directionality.",
        "random": "Random distribution demo; no dominant spatial signature.",
    }
    for p_idx, pattern in enumerate(PATTERNS):
        for i in range(count_per_pattern):
            local_seed = seed + p_idx * 1000 + i
            rng = np.random.default_rng(local_seed)
            n = int(rng.integers(28, 70))
            pts = make_points(pattern, n=n, seed=local_seed)
            size = int(rng.choice([480, 640, 800]))
            image = render_wafer(pts, size=size)
            path = out / f"{pattern}_{i:02d}_{size}px.png"
            imwrite_unicode(path, image)
            cases.append(
                (
                    path,
                    comments[pattern],
                    {"pattern": pattern, "source": "synthetic-demo"},
                )
            )
    return cases
