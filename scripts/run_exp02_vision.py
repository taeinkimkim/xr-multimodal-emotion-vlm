#!/usr/bin/env python3
"""Exp 2: Vision model inference on face images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.vlm_emotion.dataset import load_test_samples
from src.pipelines.vlm_emotion.vision_runner import Dinov2EmotionRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir", type=Path,
        default=Path("data/raw/face/balanced_rafdb/test"),
        help="Folder-organized test directory (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("models/trained/face/dinov2_balanced_rafdb/model_lora_ep10_bs16"),
        help="Path to the vision model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--backbone-pretrained-dir", type=Path,
        default=Path("models/pretrained/face/dinov2"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/face/exp02_vision"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if _cuda_available() else "cpu",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--pool-mode", choices=["cls", "pooler"], default="cls",
                        help="How to pool DINOv2 output: 'cls' (CLS token) or 'pooler' (pooler_output)")
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir / args.model_dir.parent.name / args.model_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_test_samples(args.test_dir)
    print(f"Test samples: {len(samples)}")

    runner = Dinov2EmotionRunner(
        model_dir=args.model_dir,
        backbone_pretrained_dir=args.backbone_pretrained_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=args.device,
        pool_mode=args.pool_mode,
    )

    results: list[dict] = []
    cls_tokens: list[np.ndarray] = []

    for sample in tqdm(samples, desc="Exp 2 inference"):
        out = runner.run(sample["image_path"])
        results.append({
            "image_path": sample["image_path"],
            "true_label": sample["label"],
            "predicted_label": out["predicted_label"],
            "confidence": out["confidence"],
            "probs": out["probs"],
        })
        cls_tokens.append(out["cls_token"])

    results_path = args.output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    tokens_path = args.output_dir / "cls_tokens.npy"
    np.save(tokens_path, np.stack(cls_tokens))

    _print_metrics(results, args.output_dir)
    print(f"Results saved    → {results_path}")
    print(f"CLS tokens saved → {tokens_path}")


def _print_metrics(results: list[dict], output_dir: Path) -> None:
    n = len(results)
    correct = sum(r["predicted_label"] == r["true_label"] for r in results)
    accuracy = correct / n
    print(f"\nExp 2 results — accuracy: {accuracy:.4f} ({correct}/{n})")

    metrics = {"accuracy": accuracy, "n_correct": correct, "n_total": n}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved    → {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
