"""Common command-line arguments for face emotion training."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch


def parse_face_emotion_args(
    description: str,
    default_model_name: str,
    default_output_dir: Path,
    add_model_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default="data/raw/face/affectnet")
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "affectnet", "rafdb", "generic"),
        default="auto",
        help="Dataset loader preset. Default: auto",
    )
    parser.add_argument("--model-name", default=default_model_name)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--image-root",
        default="DATASET",
        help="Image root under --dataset for datasets such as RAF-DB. Default: DATASET",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Optional JSON object or JSON file mapping raw labels to class names.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--model-filename", default="pytorch_model.bin")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--top-confused-pairs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    if add_model_args is not None:
        add_model_args(parser)
    return parser.parse_args()
