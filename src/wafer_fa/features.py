from __future__ import annotations

from collections import deque

import numpy as np

from .config import FeatureConfig
from .image_io import NormalizedWafer, normalize_points, particle_centers, segment_red_particles
from .models import FeatureBundle


def _safe_hist(values: np.ndarray, bins: int, hist_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=hist_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def radial_histogram(points: np.ndarray, bins: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(bins, dtype=np.float32)
    r = np.linalg.norm(points, axis=1)
    return _safe_hist(r, bins, (0.0, 1.0))


def angular_histogram(points: np.ndarray, bins: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(bins, dtype=np.float32)
    theta = np.arctan2(points[:, 1], points[:, 0])
    return _safe_hist(theta, bins, (-np.pi, np.pi))


def density_map(points: np.ndarray, grid: int) -> np.ndarray:
    out = np.zeros((grid, grid), dtype=np.float32)
    if len(points) == 0:
        return out
    x = np.clip(((points[:, 0] + 1.0) * 0.5 * grid).astype(int), 0, grid - 1)
    y = np.clip(((points[:, 1] + 1.0) * 0.5 * grid).astype(int), 0, grid - 1)
    for xx, yy in zip(x, y, strict=False):
        out[yy, xx] += 1.0
    total = float(out.sum())
    if total > 0:
        out /= total
    padded = np.pad(out, 1, mode="edge")
    blurred = np.zeros_like(out)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    for dy in range(3):
        for dx in range(3):
            blurred += padded[dy : dy + grid, dx : dx + grid] * kernel[dy, dx]
    s = float(blurred.sum())
    return blurred / s if s > 0 else blurred


def _pca_metrics(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    major = float(vals[order[0]])
    minor = float(vals[order[1]])
    lineality = major / (major + minor + 1e-9)
    v = vecs[:, order[0]]
    angle = float(np.arctan2(v[1], v[0]) / np.pi)
    return float(lineality), angle


def summary_features(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(12, dtype=np.float32)
    r = np.linalg.norm(points, axis=1)
    centroid = points.mean(axis=0)
    lineality, pca_angle = _pca_metrics(points)
    theta = np.arctan2(points[:, 1], points[:, 0])
    resultant = np.abs(np.mean(np.exp(1j * theta))) if len(points) else 0.0
    radial_q = np.quantile(r, [0.25, 0.50, 0.75, 0.90])
    ringness = float(np.exp(-float(np.std(r)) / 0.12)) if len(points) >= 3 else 0.0
    return np.asarray(
        [
            centroid[0],
            centroid[1],
            float(np.mean(r)),
            float(np.std(r)),
            *[float(x) for x in radial_q],
            lineality,
            pca_angle,
            float(resultant),
            ringness,
        ],
        dtype=np.float32,
    )


def _neighbors(points: np.ndarray, idx: int, eps2: float) -> np.ndarray:
    delta = points - points[idx]
    d2 = np.einsum("ij,ij->i", delta, delta)
    return np.flatnonzero(d2 <= eps2)


def cluster_features(points: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Small DBSCAN-like implementation for explainable cluster metrics."""
    n_original = len(points)
    if n_original == 0:
        return np.zeros(5, dtype=np.float32)
    pts = points
    if n_original > cfg.dbscan_max_points:
        rng = np.random.default_rng(0)
        pts = points[rng.choice(n_original, cfg.dbscan_max_points, replace=False)]
    n = len(pts)
    eps2 = cfg.dbscan_eps_norm * cfg.dbscan_eps_norm
    labels = np.full(n, -99, dtype=np.int32)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -99:
            continue
        neigh = _neighbors(pts, i, eps2)
        if len(neigh) < cfg.dbscan_min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue: deque[int] = deque(int(x) for x in neigh if int(x) != i)
        while queue:
            j = queue.popleft()
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                continue
            labels[j] = cluster_id
            neigh_j = _neighbors(pts, j, eps2)
            if len(neigh_j) >= cfg.dbscan_min_samples:
                queue.extend(int(x) for x in neigh_j if labels[int(x)] in (-99, -1))
        cluster_id += 1

    cluster_sizes = [int(np.sum(labels == c)) for c in range(cluster_id)]
    largest = max(cluster_sizes, default=0)
    clustered = int(np.sum(labels >= 0))
    noise = int(np.sum(labels == -1))
    centers = []
    for c in range(cluster_id):
        member = pts[labels == c]
        if len(member):
            centers.append(member.mean(axis=0))
    center_spread = float(np.std(np.asarray(centers), axis=0).mean()) if len(centers) >= 2 else 0.0
    return np.asarray(
        [
            min(cluster_id / 10.0, 1.0),
            largest / max(n, 1),
            clustered / max(n, 1),
            noise / max(n, 1),
            min(center_spread, 1.0),
        ],
        dtype=np.float32,
    )


def extract_features(normalized: NormalizedWafer, cfg: FeatureConfig) -> FeatureBundle:
    mask = segment_red_particles(normalized, cfg)
    points_px = particle_centers(mask, cfg)
    points = normalize_points(points_px, cfg)
    return FeatureBundle(
        particle_count=len(points),
        radial=radial_histogram(points, cfg.radial_bins),
        angular=angular_histogram(points, cfg.angular_bins),
        density=density_map(points, cfg.density_grid),
        summary=summary_features(points),
        cluster=cluster_features(points, cfg),
        normalized_points=points,
        geometry=normalized.geometry,
    )
