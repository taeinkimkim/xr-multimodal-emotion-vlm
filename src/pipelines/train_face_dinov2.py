#!/usr/bin/env python3
"""Train a DINOv2-based facial emotion classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch import nn

from src.models.face.dinov2_emotion import Dinov2EmotionClassifier
from src.pipelines.face_emotion.cli import parse_face_emotion_args
from src.pipelines.face_emotion.io import TeeLogger, make_output_filenames, resolve_output_dir
from src.pipelines.face_emotion.trainer import run_training


def parse_args() -> argparse.Namespace:
    return parse_face_emotion_args(
        description=__doc__ or "Train a DINOv2-based facial emotion classifier.",
        default_model_name="facebook/dinov2-base",
        default_output_dir=Path("models/trained/face/dinov2_emotion"),
        add_model_args=add_dinov2_args,
    )


def add_dinov2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pretrained-dir", type=Path, default=Path("models/pretrained/face/dinov2"))
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--pool-mode", choices=["cls", "pooler"], default="cls",
                        help="How to pool DINOv2 output: 'cls' (CLS token) or 'pooler' (pooler_output)")


def build_dinov2_model(args: argparse.Namespace, num_labels: int) -> nn.Module:
    return Dinov2EmotionClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        freeze_backbone=not args.unfreeze_backbone,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pretrained_dir=args.pretrained_dir,
        pool_mode=args.pool_mode,
    )


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args.output_dir, args.model_filename)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = make_output_filenames(args.model_filename)

    with TeeLogger(args.output_dir / output_files["log"]):
        run_training(args, output_files, model_factory=build_dinov2_model)


if __name__ == "__main__":
    main()
