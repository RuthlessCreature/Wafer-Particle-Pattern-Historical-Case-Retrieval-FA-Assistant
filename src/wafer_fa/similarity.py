from __future__ import annotations

import math

import numpy as np

from .config import SimilarityWeights
from .models import FeatureBundle


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def distance_score(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Feature shape mismatch: {a.shape} vs {b.shape}")
    d = float(np.sqrt(np.mean((a - b) ** 2)))
    return float(math.exp(-d / max(scale, 1e-6)))


def count_score(a: int, b: int) -> float:
    delta = abs(math.log((a + 1.0) / (b + 1.0)))
    return float(math.exp(-delta / 0.75))


def compare_features(
    query: FeatureBundle,
    candidate: FeatureBundle,
    weights: SimilarityWeights,
) -> tuple[float, dict[str, float]]:
    components = {
        "density": cosine01(query.density, candidate.density),
        "radial": cosine01(query.radial, candidate.radial),
        "angular": cosine01(query.angular, candidate.angular),
        "summary": distance_score(query.summary, candidate.summary, scale=0.30),
        "cluster": distance_score(query.cluster, candidate.cluster, scale=0.35),
        "particle_count": count_score(query.particle_count, candidate.particle_count),
    }
    wd = weights.as_dict()
    denom = sum(wd.values())
    if denom <= 0:
        raise ValueError("Similarity weights must sum to > 0")
    score = sum(components[k] * wd[k] for k in components) / denom
    return float(np.clip(score, 0.0, 1.0)), components
