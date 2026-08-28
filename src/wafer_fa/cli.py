from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig
from .service import WaferFAService
from .synthetic import generate_demo_set


def parse_meta(items: list[str] | None) -> dict[str, str]:
    meta: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Metadata must be key=value: {item}")
        key, value = item.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wafer particle historical case retrieval / FA assistant")
    parser.add_argument("--root", default=".", help="Project/data root; default current directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize data directories and SQLite database")

    add = sub.add_parser("add", help="Add one historical wafer case")
    add.add_argument("image")
    add.add_argument("--comment", default="")
    add.add_argument("--meta", action="append", help="Metadata key=value; repeatable")

    search = sub.add_parser("search", help="Find most similar historical cases")
    search.add_argument("image")
    search.add_argument("--top-k", type=int, default=3)

    sub.add_parser("rebuild", help="Re-extract all features after algorithm changes")

    demo = sub.add_parser("demo", help="Generate and ingest synthetic demo cases")
    demo.add_argument("--count-per-pattern", type=int, default=8)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = AppConfig(root=Path(args.root).resolve())
    service = WaferFAService(cfg)

    if args.command == "init":
        print(json.dumps({"db": str(cfg.db_path), "cases": service.db.count()}, ensure_ascii=False))
        return

    if args.command == "add":
        case_id = service.add_case(args.image, args.comment, parse_meta(args.meta))
        print(json.dumps({"case_id": case_id}, ensure_ascii=False))
        return

    if args.command == "search":
        rows = service.search(args.image, args.top_k)
        payload = []
        for rank, row in enumerate(rows, 1):
            payload.append(
                {
                    "rank": rank,
                    "case_id": row.case.id,
                    "score": round(row.score, 6),
                    "score_percent": round(row.score * 100, 2),
                    "image": str(row.case.image_path),
                    "comment": row.case.comment,
                    "metadata": row.case.metadata,
                    "components": {k: round(v, 6) for k, v in row.components.items()},
                }
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "rebuild":
        print(json.dumps(service.rebuild(), ensure_ascii=False))
        return

    if args.command == "demo":
        demo_dir = cfg.data_dir / "demo"
        cases = generate_demo_set(demo_dir, args.count_per_pattern)
        ids = []
        for path, comment, metadata in cases:
            ids.append(service.add_case(path, comment, metadata))
        print(json.dumps({"inserted": len(ids), "case_ids": ids}, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
