"""Copy generated result files next to the Angular app so the static site can show them.

Run after run_cls_reproduction, export_bundle and run_cls_novelty:
    python -m backend.scripts.sync_frontend_data
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ROOT / "reports" / "classification_results.json",
    ROOT / "reports" / "cls_examples.json",
    ROOT / "reports" / "cls_novelty_results.json",
    ROOT / "backend" / "models" / "cls_schema.json",
]


def main() -> None:
    dest = ROOT / "frontend" / "public"
    for src in SOURCES:
        if src.exists():
            shutil.copy2(src, dest / src.name)
            print("copied", src.name)
        else:
            print("skipped (not generated yet):", src.name)


if __name__ == "__main__":
    main()
