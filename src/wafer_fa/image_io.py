from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import FeatureConfig
from .models import WaferGeometry


class ImageReadError(RuntimeError):
    pass


@dataclass
class NormalizedWafer:
    image: np.ndarray
    mask: np.ndarray
    geometry: WaferGeometry


def imread_unicode(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(f"Unable to read image: {path}")
    return image


def imwrite_unicode(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ImageReadError(f"Unable to encode image: {path}")
    buf.tofile(str(path))


def _contour_circle(gray: np.ndarray) -> tuple[float, float, float] | None:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, float, float, float] | None = None
    image_area = float(h * w)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min(h, w) * 0.20 or radius > min(h, w) * 0.55:
            continue
        center_penalty = np.hypot(cx - w / 2, cy - h / 2) / max(h, w)
        score = circularity - 0.9 * center_penalty
        if best is None or score > best[0]:
            best = (score, cx, cy, radius)
    return None if best is None else (best[1], best[2], best[3])


def _hough_circle(gray: np.ndarray) -> tuple[float, float, float] | None:
    h, w = gray.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    work = gray
    if max_side > 1000:
        scale = 1000.0 / max_side
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blur = cv2.medianBlur(work, 7)
    hh, ww = blur.shape[:2]
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(hh, ww) * 0.3,
        param1=120,
        param2=45,
        minRadius=int(min(hh, ww) * 0.25),
        maxRadius=int(min(hh, ww) * 0.52),
    )
    if circles is None:
        return None
    candidates = circles[0]
    center = np.array([ww / 2.0, hh / 2.0])
    chosen = min(candidates, key=lambda c: float(np.linalg.norm(c[:2] - center)))
    return float(chosen[0] / scale), float(chosen[1] / scale), float(chosen[2] / scale)


def detect_wafer_circle(image: np.ndarray, cfg: FeatureConfig) -> WaferGeometry:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    contour = _contour_circle(gray)
    if contour is not None:
        return WaferGeometry(*contour, detection_method="contour")

    hough = _hough_circle(gray)
    if hough is not None:
        return WaferGeometry(*hough, detection_method="hough")

    radius = min(h, w) * cfg.wafer_radius_ratio
    return WaferGeometry(w / 2.0, h / 2.0, radius, detection_method="fallback")


def normalize_wafer(image: np.ndarray, cfg: FeatureConfig) -> NormalizedWafer:
    geom = detect_wafer_circle(image, cfg)
    size = cfg.normalized_size
    target_r = size * cfg.wafer_radius_ratio
    scale = target_r / max(geom.radius, 1.0)
    matrix = np.array(
        [
            [scale, 0.0, size / 2.0 - scale * geom.center_x],
            [0.0, scale, size / 2.0 - scale * geom.center_y],
        ],
        dtype=np.float32,
    )
    normalized = cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(target_r * 0.985), 255, -1)
    return NormalizedWafer(normalized, mask, geom)


def segment_red_particles(normalized: NormalizedWafer, cfg: FeatureConfig) -> np.ndarray:
    hsv = cv2.cvtColor(normalized.image, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(cfg.red_hsv_low_1), np.array(cfg.red_hsv_high_1))
    m2 = cv2.inRange(hsv, np.array(cfg.red_hsv_low_2), np.array(cfg.red_hsv_high_2))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.bitwise_and(mask, normalized.mask)
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def particle_centers(mask: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.empty((0, 2), dtype=np.float32)
    max_area = mask.shape[0] * mask.shape[1] * cfg.max_particle_area_ratio
    points: list[tuple[float, float]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.min_particle_area_px or area > max_area:
            continue
        cx, cy = centroids[i]
        points.append((float(cx), float(cy)))
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def normalize_points(points_px: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    if len(points_px) == 0:
        return np.empty((0, 2), dtype=np.float32)
    center = cfg.normalized_size / 2.0
    radius = cfg.normalized_size * cfg.wafer_radius_ratio
    pts = points_px.astype(np.float32).copy()
    pts[:, 0] = (pts[:, 0] - center) / radius
    pts[:, 1] = (pts[:, 1] - center) / radius
    inside = np.sum(pts * pts, axis=1) <= 1.0
    return pts[inside]
