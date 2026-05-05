#!/usr/bin/env python3
"""Download an AffectNet dataset repository from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "face" / "affectnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AffectNet files from a Hugging Face dataset repo."
    )
    parser.add_argument(
        "repo_id",
        help=(
            "Hugging Face dataset repo id, for example "
            "'owner/affectnet'. Copy this from the dataset URL."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset branch, tag, or commit SHA. Default: main",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help=(
            "Only download matching files. Can be repeated, e.g. "
            "--allow-pattern '*.parquet' --allow-pattern '*.json'"
        ),
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help="Skip matching files. Can be repeated.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent file downloads. Default: 1, which is safer "
            "for low-memory machines."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=output_dir,
        token=args.token,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        max_workers=args.max_workers,
    )

    print(f"Downloaded {args.repo_id} to {downloaded_path}")


if __name__ == "__main__":
    main()
