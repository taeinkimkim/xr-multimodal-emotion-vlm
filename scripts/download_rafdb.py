#!/usr/bin/env python3
"""Download RAF-DB from Kaggle using kagglehub."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "face" / "rafdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RAF-DB files from Kaggle with kagglehub."
    )
    parser.add_argument(
        "dataset_id",
        help=(
            "Kaggle dataset id, for example 'owner/raf-db-dataset'. "
            "Copy this from the dataset URL."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download again even if the dataset already exists in output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required. Install it with: pip install kagglehub"
        ) from exc

    downloaded_path = Path(
        kagglehub.dataset_download(
            args.dataset_id,
            output_dir=str(output_dir),
            force_download=args.force_download,
        )
    ).resolve()
    print(f"Downloaded {args.dataset_id} to {downloaded_path}")


if __name__ == "__main__":
    main()
