"""End-to-end face emotion training orchestration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, get_cosine_schedule_with_warmup

from src.data.face.emotion_dataset import (
    FaceEmotionColumns,
    FaceEmotionTorchDataset,
    load_face_emotion_dataset,
    parse_label_map,
)
from src.pipelines.face_emotion.epoch import run_epoch
from src.pipelines.face_emotion.io import save_checkpoint, save_json
from src.pipelines.face_emotion.metrics import prefix_metrics, print_evaluation_metrics


ModelFactory = Callable[[argparse.Namespace, int], nn.Module]


def run_training(
    args: argparse.Namespace,
    output_files: dict[str, str],
    model_factory: ModelFactory,
) -> None:
    print(f"Model file: {args.output_dir / output_files['model']}")
    print(f"Metrics file: {args.output_dir / output_files['metrics']}")
    print(f"Best model file: {args.output_dir / output_files['best_model']}")
    print(f"Best metrics file: {args.output_dir / output_files['best_metrics']}")
    print(f"History file: {args.output_dir / output_files['history']}")
    print(f"Log file: {args.output_dir / output_files['log']}")

    processor = load_image_processor(args.model_name, getattr(args, "pretrained_dir", None))
    dataset, label2id, id2label = load_face_emotion_dataset(
        args.dataset,
        dataset_type=args.dataset_type,
        image_column=args.image_column,
        label_column=args.label_column,
        image_root=args.image_root,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        label_map=parse_label_map(args.label_map),
    )

    if args.train_split not in dataset:
        raise ValueError(f"Missing train split {args.train_split!r}. Found: {list(dataset)}")
    eval_split = args.eval_split if args.eval_split in dataset else None
    test_split = args.test_split if args.test_split in dataset else None

    print(f"Dataset splits: {', '.join(f'{split}={len(split_dataset)}' for split, split_dataset in dataset.items())}")
    print(f"Labels: {label2id}")

    columns = FaceEmotionColumns(image=args.image_column, label=args.label_column)
    train_dataset = FaceEmotionTorchDataset(dataset[args.train_split], processor, columns, label2id)
    eval_dataset = FaceEmotionTorchDataset(dataset[eval_split], processor, columns, label2id) if eval_split else None
    test_dataset = FaceEmotionTorchDataset(dataset[test_split], processor, columns, label2id) if test_split else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers) if eval_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) if test_dataset else None

    model = model_factory(args, len(id2label)).to(args.device)

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
        print(f"\n===========================================================")
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
                _save_current_and_best_checkpoints(args, output_files, model, processor, label2id, id2label, metrics)
        else:
            best_state_dict = _clone_state_dict(model)
            best_metrics = metrics
            _save_current_and_best_checkpoints(args, output_files, model, processor, label2id, id2label, metrics)

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
        _save_current_and_best_checkpoints(args, output_files, model, processor, label2id, id2label, metrics)
        final_metrics = metrics

    history_report = {"epochs": history, "final": final_metrics or {}}
    save_json(args.output_dir / output_files["history"], history_report)
    print(f"Saved training history to {args.output_dir / output_files['history']}")


def _save_current_and_best_checkpoints(
    args: argparse.Namespace,
    output_files: dict[str, str],
    model: nn.Module,
    processor: object,
    label2id: dict[str, int],
    id2label: dict[int, str],
    metrics: dict[str, object],
) -> None:
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


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def load_image_processor(model_name: str, pretrained_dir: Path | None) -> object:
    if pretrained_dir is None:
        return AutoImageProcessor.from_pretrained(model_name)

    pretrained_dir.mkdir(parents=True, exist_ok=True)
    processor_source = pretrained_dir if (pretrained_dir / "preprocessor_config.json").exists() else model_name
    processor = AutoImageProcessor.from_pretrained(
        processor_source,
        cache_dir=str(pretrained_dir),
    )
    if processor_source == model_name:
        processor.save_pretrained(pretrained_dir)
    return processor
