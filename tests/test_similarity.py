from pathlib import Path

from wafer_fa.config import AppConfig
from wafer_fa.image_io import imwrite_unicode
from wafer_fa.service import WaferFAService
from wafer_fa.synthetic import make_points, render_wafer


def test_same_pattern_beats_different_pattern(tmp_path: Path) -> None:
    service = WaferFAService(AppConfig(root=tmp_path))

    line_a = tmp_path / "line_a.png"
    line_b = tmp_path / "line_b.png"
    random_a = tmp_path / "random.png"

    imwrite_unicode(line_a, render_wafer(make_points("line", 55, seed=10), size=480))
    imwrite_unicode(line_b, render_wafer(make_points("line", 60, seed=11), size=800))
    imwrite_unicode(random_a, render_wafer(make_points("random", 55, seed=12), size=640))

    line_id = service.add_case(line_a, "line", {"pattern": "line"})
    service.add_case(random_a, "random", {"pattern": "random"})

    results = service.search(line_b, top_k=2)
    assert results[0].case.id == line_id
    assert results[0].score > results[1].score


def test_scale_robustness(tmp_path: Path) -> None:
    service = WaferFAService(AppConfig(root=tmp_path))
    points = make_points("center", 45, seed=77)
    a = tmp_path / "center_480.png"
    b = tmp_path / "center_900.png"
    imwrite_unicode(a, render_wafer(points, size=480))
    imwrite_unicode(b, render_wafer(points, size=900))
    service.add_case(a, "same point set")
    result = service.search(b, top_k=1)[0]
    assert result.score > 0.88
