#!/usr/bin/env python3
"""Download an AffectNet dataset from Hugging Face Hub or Kaggle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "face" / "affectnet"
DEFAULT_KAGGLE_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "face" / "balanced_affectnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AffectNet files from Hugging Face Hub or Kaggle."
    )
    parser.add_argument(
        "dataset_id",
        help=(
            "Dataset id. For Hugging Face, use a dataset repo id like "
            "'owner/affectnet'. For Kaggle, use a dataset id like "
            "'owner/dataset-name'. Copy this from the dataset URL."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "kaggle"),
        default="huggingface",
        help="Where to download from. Default: huggingface",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Where to save files. Defaults to {DEFAULT_HF_OUTPUT_DIR} for "
            f"Hugging Face and {DEFAULT_KAGGLE_OUTPUT_DIR} for Kaggle."
        ),
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
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="For Kaggle downloads, download again even if output-dir already has the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "kaggle":
        downloaded_path = download_from_kaggle(args, output_dir)
    else:
        downloaded_path = download_from_huggingface(args, output_dir)

    print(f"Downloaded {args.dataset_id} to {downloaded_path}")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    if args.source == "kaggle":
        return DEFAULT_KAGGLE_OUTPUT_DIR.resolve()
    return DEFAULT_HF_OUTPUT_DIR.resolve()


def download_from_huggingface(args: argparse.Namespace, output_dir: Path) -> str:
    return snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=output_dir,
        token=args.token,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        max_workers=args.max_workers,
    )


def download_from_kaggle(args: argparse.Namespace, output_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required. Install it with: pip install kagglehub"
        ) from exc

    return Path(
        kagglehub.dataset_download(
            args.dataset_id,
            output_dir=str(output_dir),
            force_download=args.force_download,
        )
    ).resolve()


if __name__ == "__main__":
    main()
