#!/usr/bin/env python3
"""Train a DINOv2-based AffectNet facial emotion classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, get_cosine_schedule_with_warmup

from src.data.face.affectnet_dataset import (
    AffectNetColumns,
    AffectNetTorchDataset,
    load_affectnet_dataset,
)
from src.models.face.dinov2_emotion import Dinov2EmotionClassifier, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/raw/face/affectnet")
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models/trained/face/dinov2_affectnet"))
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
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = make_output_filenames(args.model_filename)

    with TeeLogger(args.output_dir / output_files["log"]):
        run_training(args, output_files)


def run_training(args: argparse.Namespace, output_files: dict[str, str]) -> None:
    print(f"Model file: {args.output_dir / output_files['model']}")
    print(f"Metrics file: {args.output_dir / output_files['metrics']}")
    print(f"Best model file: {args.output_dir / output_files['best_model']}")
    print(f"Best metrics file: {args.output_dir / output_files['best_metrics']}")
    print(f"History file: {args.output_dir / output_files['history']}")
    print(f"Log file: {args.output_dir / output_files['log']}")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    dataset, label2id, id2label = load_affectnet_dataset(
        args.dataset,
        image_column=args.image_column,
        label_column=args.label_column,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
    )

    if args.train_split not in dataset:
        raise ValueError(f"Missing train split {args.train_split!r}. Found: {list(dataset)}")
    eval_split = args.eval_split if args.eval_split in dataset else None
    test_split = args.test_split if args.test_split in dataset else None

    columns = AffectNetColumns(image=args.image_column, label=args.label_column)
    train_dataset = AffectNetTorchDataset(dataset[args.train_split], processor, columns, label2id)
    eval_dataset = (
        AffectNetTorchDataset(dataset[eval_split], processor, columns, label2id) if eval_split else None
    )
    test_dataset = (
        AffectNetTorchDataset(dataset[test_split], processor, columns, label2id) if test_split else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    eval_loader = (
        DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        if eval_dataset
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        if test_dataset
        else None
    )

    model = Dinov2EmotionClassifier(
        model_name=args.model_name,
        num_labels=len(id2label),
        freeze_backbone=not args.unfreeze_backbone,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(args.device)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    final_metrics: dict[str, object] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=args.device,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            description=f"epoch {epoch}/{args.epochs} train",
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }
        if eval_loader:
            eval_metrics = run_epoch(
                model=model,
                loader=eval_loader,
                criterion=criterion,
                device=args.device,
                train=False,
                description=f"epoch {epoch}/{args.epochs} eval",
                id2label=id2label,
                collect_class_metrics=True,
                ece_bins=args.ece_bins,
                top_confused_pairs=args.top_confused_pairs,
            )
            metrics.update(prefix_metrics(eval_metrics, prefix="eval"))
            print_evaluation_metrics(eval_metrics, split_name="eval")

            if eval_metrics["accuracy"] >= best_accuracy:
                best_accuracy = eval_metrics["accuracy"]
                best_state_dict = _clone_state_dict(model)
                best_metrics = metrics
                save_checkpoint(
                    args.output_dir,
                    model,
                    processor,
                    label2id,
                    id2label,
                    metrics,
                    model_filename=output_files["model"],
                    metrics_filename=output_files["metrics"],
                )
                save_checkpoint(
                    args.output_dir,
                    model,
                    processor,
                    label2id,
                    id2label,
                    metrics,
                    model_filename=output_files["best_model"],
                    metrics_filename=output_files["best_metrics"],
                )
        else:
            best_state_dict = _clone_state_dict(model)
            best_metrics = metrics
            save_checkpoint(
                args.output_dir,
                model,
                processor,
                label2id,
                id2label,
                metrics,
                model_filename=output_files["model"],
                metrics_filename=output_files["metrics"],
            )
            save_checkpoint(
                args.output_dir,
                model,
                processor,
                label2id,
                id2label,
                metrics,
                model_filename=output_files["best_model"],
                metrics_filename=output_files["best_metrics"],
            )

        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        history.append(metrics)
        final_metrics = metrics

    if test_loader:
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=args.device,
            train=False,
            description="test",
            id2label=id2label,
            collect_class_metrics=True,
            ece_bins=args.ece_bins,
            top_confused_pairs=args.top_confused_pairs,
        )
        metrics = {
            **(best_metrics or {}),
            **prefix_metrics(test_metrics, prefix="test"),
        }
        print_evaluation_metrics(test_metrics, split_name="test")
        save_checkpoint(
            args.output_dir,
            model,
            processor,
            label2id,
            id2label,
            metrics,
            model_filename=output_files["model"],
            metrics_filename=output_files["metrics"],
        )
        save_checkpoint(
            args.output_dir,
            model,
            processor,
            label2id,
            id2label,
            metrics,
            model_filename=output_files["best_model"],
            metrics_filename=output_files["best_metrics"],
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        final_metrics = metrics

    history_report = {"epochs": history, "final": final_metrics or {}}
    save_json(args.output_dir / output_files["history"], history_report)
    print(f"Saved training history to {args.output_dir / output_files['history']}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    train: bool,
    description: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    id2label: dict[int, str] | None = None,
    collect_class_metrics: bool = False,
    ece_bins: int = 15,
    top_confused_pairs: int = 10,
) -> dict[str, object]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    class_stats = (
        _new_evaluation_stats(len(id2label or {}), ece_bins) if collect_class_metrics else None
    )

    for batch in tqdm(loader, desc=description):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(pixel_values)
            loss = criterion(logits, labels)

        if train:
            if optimizer is None or scheduler is None:
                raise ValueError("optimizer and scheduler are required for training")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

        if class_stats is not None:
            sample_losses = F.cross_entropy(logits, labels, reduction="none")
            probabilities = logits.softmax(dim=-1)
            _update_class_stats(
                stats=class_stats,
                labels=labels,
                predictions=predictions,
                probabilities=probabilities,
                sample_losses=sample_losses,
            )

    metrics: dict[str, object] = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
    if class_stats is not None:
        if id2label is None:
            raise ValueError("id2label is required to collect class metrics")
        metrics.update(
            _finalize_evaluation_metrics(
                class_stats,
                id2label,
                top_confused_pairs=top_confused_pairs,
            )
        )
    return metrics


def _new_evaluation_stats(num_labels: int, ece_bins: int) -> dict[str, torch.Tensor]:
    return {
        "support": torch.zeros(num_labels, dtype=torch.long),
        "correct": torch.zeros(num_labels, dtype=torch.long),
        "loss_sum": torch.zeros(num_labels, dtype=torch.float64),
        "true_probability_sum": torch.zeros(num_labels, dtype=torch.float64),
        "predicted_probability_sum": torch.zeros(num_labels, dtype=torch.float64),
        "softmax_sum": torch.zeros(num_labels, num_labels, dtype=torch.float64),
        "confusion_matrix": torch.zeros(num_labels, num_labels, dtype=torch.long),
        "confidence_sum": torch.zeros(1, dtype=torch.float64),
        "correct_confidence_sum": torch.zeros(1, dtype=torch.float64),
        "incorrect_confidence_sum": torch.zeros(1, dtype=torch.float64),
        "correct_count": torch.zeros(1, dtype=torch.long),
        "incorrect_count": torch.zeros(1, dtype=torch.long),
        "ece_confidence_sum": torch.zeros(ece_bins, dtype=torch.float64),
        "ece_correct_sum": torch.zeros(ece_bins, dtype=torch.float64),
        "ece_count": torch.zeros(ece_bins, dtype=torch.long),
    }


def _update_class_stats(
    stats: dict[str, torch.Tensor],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    sample_losses: torch.Tensor,
) -> None:
    labels_cpu = labels.detach().cpu()
    predictions_cpu = predictions.detach().cpu()
    probabilities_cpu = probabilities.detach().cpu().to(torch.float64)
    sample_losses_cpu = sample_losses.detach().cpu().to(torch.float64)

    for index, label in enumerate(labels_cpu):
        label_id = int(label.item())
        prediction_id = int(predictions_cpu[index].item())
        stats["support"][label_id] += 1
        stats["correct"][label_id] += int(prediction_id == label_id)
        stats["loss_sum"][label_id] += sample_losses_cpu[index]
        stats["true_probability_sum"][label_id] += probabilities_cpu[index, label_id]
        stats["predicted_probability_sum"][label_id] += probabilities_cpu[index, prediction_id]
        stats["softmax_sum"][label_id] += probabilities_cpu[index]
        stats["confusion_matrix"][label_id, prediction_id] += 1

        confidence = probabilities_cpu[index, prediction_id]
        is_correct = prediction_id == label_id
        stats["confidence_sum"][0] += confidence
        if is_correct:
            stats["correct_confidence_sum"][0] += confidence
            stats["correct_count"][0] += 1
        else:
            stats["incorrect_confidence_sum"][0] += confidence
            stats["incorrect_count"][0] += 1

        ece_bins = stats["ece_count"].numel()
        bin_index = min(int(confidence.item() * ece_bins), ece_bins - 1)
        stats["ece_confidence_sum"][bin_index] += confidence
        stats["ece_correct_sum"][bin_index] += float(is_correct)
        stats["ece_count"][bin_index] += 1


def _finalize_evaluation_metrics(
    stats: dict[str, torch.Tensor],
    id2label: dict[int, str],
    top_confused_pairs: int,
) -> dict[str, object]:
    per_class_metrics: dict[str, dict[str, object]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    weighted_f1_sum = 0.0
    total_support = int(stats["support"].sum().item())

    for label_id in sorted(id2label):
        label = id2label[label_id]
        support = int(stats["support"][label_id].item())
        if support == 0:
            continue

        true_positive = int(stats["confusion_matrix"][label_id, label_id].item())
        predicted_positive = int(stats["confusion_matrix"][:, label_id].sum().item())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)
        weighted_f1_sum += f1 * support

        per_class_metrics[label] = {
            "support": support,
            "loss": float(stats["loss_sum"][label_id].item() / support),
            "accuracy": float(stats["correct"][label_id].item() / support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_true_class_softmax": float(
                stats["true_probability_sum"][label_id].item() / support
            ),
            "avg_predicted_softmax": float(
                stats["predicted_probability_sum"][label_id].item() / support
            ),
            "mean_softmax": {
                id2label[index]: float(value.item() / support)
                for index, value in enumerate(stats["softmax_sum"][label_id])
            },
        }
    confusion_matrix = _format_confusion_matrix(stats["confusion_matrix"], id2label)
    return {
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "weighted_f1": weighted_f1_sum / total_support if total_support else 0.0,
        "mean_confidence": _safe_div(
            float(stats["confidence_sum"][0].item()), total_support
        ),
        "correct_confidence": _safe_div(
            float(stats["correct_confidence_sum"][0].item()),
            int(stats["correct_count"][0].item()),
        ),
        "incorrect_confidence": _safe_div(
            float(stats["incorrect_confidence_sum"][0].item()),
            int(stats["incorrect_count"][0].item()),
        ),
        "ece": _compute_ece(stats),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "top_confused_pairs": _top_confused_pairs(
            stats["confusion_matrix"],
            id2label,
            top_k=top_confused_pairs,
        ),
    }


def _safe_div(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_ece(stats: dict[str, torch.Tensor]) -> float:
    total = int(stats["ece_count"].sum().item())
    if total == 0:
        return 0.0

    ece = 0.0
    for index, count_tensor in enumerate(stats["ece_count"]):
        count = int(count_tensor.item())
        if count == 0:
            continue
        average_confidence = float(stats["ece_confidence_sum"][index].item() / count)
        average_accuracy = float(stats["ece_correct_sum"][index].item() / count)
        ece += (count / total) * abs(average_accuracy - average_confidence)
    return ece


def _format_confusion_matrix(
    confusion_matrix: torch.Tensor,
    id2label: dict[int, str],
) -> dict[str, dict[str, int]]:
    return {
        id2label[row_index]: {
            id2label[column_index]: int(value.item())
            for column_index, value in enumerate(row)
        }
        for row_index, row in enumerate(confusion_matrix)
    }


def _top_confused_pairs(
    confusion_matrix: torch.Tensor,
    id2label: dict[int, str],
    top_k: int,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for true_index, row in enumerate(confusion_matrix):
        true_count = int(row.sum().item())
        for predicted_index, value in enumerate(row):
            if true_index == predicted_index:
                continue
            count = int(value.item())
            if count == 0:
                continue
            pairs.append(
                {
                    "true_label": id2label[true_index],
                    "predicted_label": id2label[predicted_index],
                    "count": count,
                    "rate_within_true_label": count / true_count if true_count else 0.0,
                }
            )
    return sorted(pairs, key=lambda pair: pair["count"], reverse=True)[:top_k]


def prefix_metrics(metrics: dict[str, object], prefix: str) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def print_evaluation_metrics(metrics: dict[str, object], split_name: str) -> None:
    print(f"{split_name} metrics")
    print(
        "  "
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.2%} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.2%} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"mean_confidence={metrics['mean_confidence']:.4f} "
        f"correct_confidence={metrics['correct_confidence']:.4f} "
        f"incorrect_confidence={metrics['incorrect_confidence']:.4f} "
        f"ece={metrics['ece']:.4f}"
    )
    print(f"{split_name} per-class metrics")
    print("label      support  precision  recall  f1      accuracy  loss")
    print("---------  -------  ---------  ------  ------  --------  ------")
    for label, class_metrics in metrics["per_class_metrics"].items():
        print(
            f"{label:<9}  "
            f"{class_metrics['support']:>7}  "
            f"{class_metrics['precision']:>9.2%}  "
            f"{class_metrics['recall']:>6.2%}  "
            f"{class_metrics['f1']:>6.4f}  "
            f"{class_metrics['accuracy']:>8.2%}  "
            f"{class_metrics['loss']:>6.4f}"
        )
    print_confusion_matrix(metrics["confusion_matrix"], split_name=split_name)
    print(f"{split_name} top confused pairs")
    for pair in metrics["top_confused_pairs"]:
        print(
            "  "
            f"{pair['true_label']} -> {pair['predicted_label']}: "
            f"{pair['count']} ({pair['rate_within_true_label']:.2%})"
        )


def print_confusion_matrix(
    confusion_matrix: dict[str, dict[str, int]],
    split_name: str,
) -> None:
    labels = list(confusion_matrix)
    column_width = max(7, max(len(label) for label in labels))
    print(f"{split_name} confusion matrix")
    print("true\\pred".ljust(column_width), end="")
    for label in labels:
        print(f"  {label:>{column_width}}", end="")
    print()
    for true_label in labels:
        print(f"{true_label:<{column_width}}", end="")
        for predicted_label in labels:
            print(f"  {confusion_matrix[true_label][predicted_label]:>{column_width}}", end="")
        print()


def make_output_filenames(model_filename: str) -> dict[str, str]:
    model_path = Path(model_filename)
    if model_path.name != model_filename:
        raise ValueError("--model-filename must be a filename, not a path")

    stem = model_path.stem
    suffix = model_path.suffix or ".bin"
    model = model_path.name if model_path.suffix else f"{model_path.name}{suffix}"
    return {
        "model": model,
        "metrics": f"{stem}_metrics.json",
        "log": f"{stem}_log.txt",
        "history": f"{stem}_history.json",
        "best_model": f"{stem}_best{suffix}",
        "best_metrics": f"{stem}_best_metrics.json",
    }


def save_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


class TeeLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.file: TextIO | None = None
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None

    def __enter__(self) -> "TeeLogger":
        self.file = self.log_path.open("w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = _TeeStream(self.stdout, self.file)
        sys.stderr = _TeeStream(self.stderr, self.file)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.stdout is not None:
            sys.stdout = self.stdout
        if self.stderr is not None:
            sys.stderr = self.stderr
        if self.file is not None:
            self.file.close()


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(
    output_dir: Path,
    model: Dinov2EmotionClassifier,
    processor: AutoImageProcessor,
    label2id: dict[str, int],
    id2label: dict[int, str],
    metrics: dict[str, object],
    model_filename: str = "pytorch_model.bin",
    metrics_filename: str = "metrics.json",
) -> None:
    torch.save(model.state_dict(), output_dir / model_filename)
    processor.save_pretrained(output_dir)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            file,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / metrics_filename).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
