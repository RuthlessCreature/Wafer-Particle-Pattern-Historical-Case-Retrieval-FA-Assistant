from pathlib import Path

from wafer_fa.config import AppConfig
from wafer_fa.image_io import imwrite_unicode
from wafer_fa.service import WaferFAService
from wafer_fa.synthetic import generate_demo_set, make_points, render_wafer


def main() -> None:
    root = Path.cwd()
    service = WaferFAService(AppConfig(root=root))
    demo_dir = root / "data" / "demo"
    if service.db.count() == 0:
        for path, comment, meta in generate_demo_set(demo_dir, count_per_pattern=6):
            service.add_case(path, comment, meta)

    query = demo_dir / "query_line.png"
    imwrite_unicode(query, render_wafer(make_points("line", 55, seed=999), size=720))
    for rank, result in enumerate(service.search(query, top_k=3), 1):
        print(rank, round(result.score * 100, 1), result.case.metadata, result.case.comment)


if __name__ == "__main__":
    main()
