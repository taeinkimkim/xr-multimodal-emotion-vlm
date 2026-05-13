#!/usr/bin/env python3
"""Download google/gemma-4-E4B-it in bfloat16 (no quantization) and save to disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vlm.gemma4 import MODEL_ID, SAVE_SUBDIR_BF16, download_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        metavar="HF_ID",
        help="Hugging Face model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/pretrained/vlm") / SAVE_SUBDIR_BF16,
        help="Directory to save the model (default: %(default)s)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map, e.g. 'auto' or 'cuda:0' (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_save(
        model_id=args.model_id,
        output_dir=args.output_dir,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
