from __future__ import annotations

import io

import numpy as np

from .models import FeatureBundle


def pack_features(bundle: FeatureBundle) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(
        bio,
        particle_count=np.asarray([bundle.particle_count], dtype=np.int32),
        radial=bundle.radial.astype(np.float32),
        angular=bundle.angular.astype(np.float32),
        density=bundle.density.astype(np.float32),
        summary=bundle.summary.astype(np.float32),
        cluster=bundle.cluster.astype(np.float32),
        normalized_points=bundle.normalized_points.astype(np.float32),
    )
    return bio.getvalue()


def unpack_features(blob: bytes) -> FeatureBundle:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return FeatureBundle(
            particle_count=int(data["particle_count"][0]),
            radial=data["radial"].astype(np.float32),
            angular=data["angular"].astype(np.float32),
            density=data["density"].astype(np.float32),
            summary=data["summary"].astype(np.float32),
            cluster=data["cluster"].astype(np.float32),
            normalized_points=data["normalized_points"].astype(np.float32),
            geometry=None,
        )
