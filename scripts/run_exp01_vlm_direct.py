#!/usr/bin/env python3
"""Exp 1: VLM direct inference on face images — no vision model input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.vlm_emotion.dataset import load_test_samples
from src.pipelines.vlm_emotion.vlm_runner import Gemma4Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir", type=Path,
        default=Path("data/raw/face/balanced_rafdb/test"),
        help="Folder-organized test directory (default: %(default)s)",
    )
    parser.add_argument(
        "--vlm-model-dir", type=Path,
        default=Path("models/pretrained/vlm/gemma-4-E4B-it-4bit"),
        help="Path to the VLM model (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/face/exp01_vlm_direct"),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prompt-id", type=int, choices=[0, 1, 2], default=2,
                        help="Prompt template to use (0=no-reasoning, 1=simple, 2=step-by-step; default: 2)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode in apply_chat_template")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def _load_partial(results_path: Path) -> list[dict]:
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir / args.vlm_model_dir.name / f"prompt_id_{args.prompt_id}"
    if args.enable_thinking:
        args.output_dir = args.output_dir / "enable_thinking"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"

    samples = load_test_samples(args.test_dir)
    print(f"Test samples: {len(samples)}")

    done = _load_partial(results_path)
    done_paths = {r["image_path"] for r in done}
    remaining = [s for s in samples if s["image_path"] not in done_paths]
    print(f"Already done: {len(done)}  |  Remaining: {len(remaining)}")

    if not remaining:
        print("All samples already processed.")
    else:
        runner = Gemma4Runner(
            model_dir=args.vlm_model_dir,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
            prompt_id=args.prompt_id,
            enable_thinking=args.enable_thinking,
        )

        results = list(done)
        for sample in tqdm(remaining, desc="Exp 1 inference"):
            out = runner.run_direct(sample["image_path"])
            results.append({
                "image_path": sample["image_path"],
                "true_label": sample["label"],
                "predicted_label": out["predicted_label"],
                "response": out["response"],
            })
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = _load_partial(results_path)
    _print_metrics(results, args.output_dir)


def _print_metrics(results: list[dict], output_dir: Path) -> None:
    n = len(results)
    if n == 0:
        return
    correct = sum(r.get("predicted_label") == r["true_label"] for r in results)
    unparsed = sum(r.get("predicted_label") is None for r in results)
    accuracy = correct / n
    print(f"\nExp 1 results — accuracy: {accuracy:.4f} ({correct}/{n})  unparsed: {unparsed}")

    metrics = {"accuracy": accuracy, "n_correct": correct, "n_total": n, "n_unparsed": unparsed}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
