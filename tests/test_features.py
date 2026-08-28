from pathlib import Path

import numpy as np

from wafer_fa.config import AppConfig
from wafer_fa.image_io import imwrite_unicode
from wafer_fa.service import WaferFAService
from wafer_fa.synthetic import make_points, render_wafer


def test_feature_shapes_and_particle_detection(tmp_path: Path) -> None:
    service = WaferFAService(AppConfig(root=tmp_path))
    path = tmp_path / "ring.png"
    imwrite_unicode(path, render_wafer(make_points("ring", 50, seed=1), size=640))
    features, normalized = service.analyze(path)

    assert features.particle_count >= 40
    assert features.radial.shape == (20,)
    assert features.angular.shape == (36,)
    assert features.density.shape == (16, 16)
    assert features.summary.shape == (12,)
    assert features.cluster.shape == (5,)
    assert normalized.image.shape[:2] == (512, 512)
    assert np.isclose(features.radial.sum(), 1.0, atol=1e-5)
