"""Fetch BDG2 from Kaggle and build the canonical parquet panel.

Usage: python -m backend.scripts.fetch_data [--cohort novelty|reproduction] [--n 120]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub

from backend.ml.data import PROCESSED_DIR, RAW_DIR, build_panel, write_panel


NEEDED = ("metadata.csv", "weather.csv", "electricity.csv")


def fetch_raw(target: Path = RAW_DIR) -> Path:
    downloaded = Path(kagglehub.dataset_download("claytonmiller/buildingdatagenomeproject2"))
    target.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve() != target.resolve():
        for name in NEEDED:
            source = downloaded / name
            destination = target / name
            if not source.exists():
                raise FileNotFoundError(f"{name} missing from Kaggle download")
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() and destination.resolve() == source.resolve():
                    continue
                if not destination.is_symlink() and destination.stat().st_size == source.stat().st_size:
                    continue
                destination.unlink()
            # Symlink instead of copying 713 MB; data/raw is gitignored scratch.
            destination.symlink_to(source)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["novelty", "reproduction"], default="novelty")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    raw_dir = fetch_raw()
    panel = build_panel(raw_dir, cohort=args.cohort, n=args.n)
    path = write_panel(panel)
    print(f"rows={len(panel):,}")
    print(f"buildings={panel['building_id'].nunique():,}")
    print(f"sites={panel['site_id'].nunique():,}")
    print(f"panel={path} ({path.stat().st_size / 1e6:.1f} MB)")
    if not args.keep_raw:
        print(f"raw cache retained at {RAW_DIR} (delete manually if disk is tight)")


if __name__ == "__main__":
    main()
