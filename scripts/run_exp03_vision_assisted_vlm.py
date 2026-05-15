#!/usr/bin/env python3
"""Exp 3: Vision model's prediction fed into VLM as context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.vlm_emotion.dataset import load_test_samples
from src.pipelines.vlm_emotion.vlm_runner import Gemma4Runner
from src.pipelines.vlm_emotion.vision_runner import Dinov2EmotionRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir", type=Path,
        default=Path("data/raw/face/balanced_rafdb/test"),
    )
    parser.add_argument(
        "--vision-model-dir", type=Path,
        default=Path("models/trained/face/dinov2_balanced_rafdb/model_lora_ep10_bs16"),
    )
    parser.add_argument(
        "--vlm-model-dir", type=Path,
        default=Path("models/pretrained/vlm/gemma-4-E4B-it-4bit"),
    )
    parser.add_argument(
        "--backbone-pretrained-dir", type=Path,
        default=Path("models/pretrained/face/dinov2"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/face/exp03_vision_assisted_vlm"),
    )
    parser.add_argument(
        "--vision-device",
        default="cuda" if _cuda_available() else "cpu",
    )
    parser.add_argument("--vlm-device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
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


def _load_partial(results_path: Path) -> list[dict]:
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def _load_partial_tokens(tokens_path: Path) -> list[np.ndarray]:
    if tokens_path.exists():
        arr = np.load(tokens_path)
        return [arr[i] for i in range(len(arr))]
    return []


def main() -> None:
    args = parse_args()
    args.output_dir = (
        args.output_dir
        / f"{args.vision_model_dir.parent.name}+{args.vlm_model_dir.name}"
        / args.vision_model_dir.name
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "results.json"
    tokens_path = args.output_dir / "cls_tokens.npy"

    samples = load_test_samples(args.test_dir)
    print(f"Test samples: {len(samples)}")

    done = _load_partial(results_path)
    done_tokens = _load_partial_tokens(tokens_path)
    done_paths = {r["image_path"] for r in done}
    remaining = [s for s in samples if s["image_path"] not in done_paths]
    print(f"Already done: {len(done)}  |  Remaining: {len(remaining)}")

    if remaining:
        vision_runner = Dinov2EmotionRunner(
            model_dir=args.vision_model_dir,
            backbone_pretrained_dir=args.backbone_pretrained_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device=args.vision_device,
            pool_mode=args.pool_mode,
        )
        gemma4_runner = Gemma4Runner(
            model_dir=args.vlm_model_dir,
            device_map=args.vlm_device_map,
            max_new_tokens=args.max_new_tokens,
        )

        results = list(done)
        cls_tokens = list(done_tokens)

        for sample in tqdm(remaining, desc="Exp 3 inference"):
            vision_out = vision_runner.run(sample["image_path"])
            vision_label = vision_out["predicted_label"]

            gemma_out = gemma4_runner.run_vision_assisted(
                sample["image_path"],
                vision_label=vision_label,
                cls_token=vision_out["cls_token"],
            )

            results.append({
                "image_path": sample["image_path"],
                "true_label": sample["label"],
                "vision_predicted_label": vision_label,
                "vision_confidence": vision_out["confidence"],
                "predicted_label": gemma_out["predicted_label"],
                "response": gemma_out["response"],
            })
            cls_tokens.append(vision_out["cls_token"])

            # Save after every sample (resumable)
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            np.save(tokens_path, np.stack(cls_tokens))

        del vision_runner, gemma4_runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = _load_partial(results_path)
    _print_metrics(results, args.output_dir)
    print(f"Results saved    → {results_path}")
    print(f"CLS tokens saved → {tokens_path}")


def _print_metrics(results: list[dict], output_dir: Path) -> None:
    n = len(results)
    if n == 0:
        return

    gemma_correct = sum(r.get("predicted_label") == r["true_label"] for r in results)
    vision_correct = sum(r.get("vision_predicted_label") == r["true_label"] for r in results)
    unparsed = sum(r.get("predicted_label") is None for r in results)

    print(f"\nExp 3 results ({n} samples)")
    print(f"  Vision accuracy : {vision_correct / n:.4f} ({vision_correct}/{n})")
    print(f"  Gemma 4 accuracy: {gemma_correct / n:.4f} ({gemma_correct}/{n})  unparsed: {unparsed}")

    metrics = {
        "gemma4_accuracy": gemma_correct / n,
        "vision_accuracy": vision_correct / n,
        "n_correct_gemma4": gemma_correct,
        "n_correct_vision": vision_correct,
        "n_total": n,
        "n_unparsed": unparsed,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved    → {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
