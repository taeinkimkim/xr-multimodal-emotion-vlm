#!/usr/bin/env python3
"""Train a DINOv2-based AffectNet facial emotion classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
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
    parser.add_argument("--output-dir", type=Path, default=Path("models/face/dinov2_affectnet"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
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

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    dataset, label2id, id2label = load_affectnet_dataset(
        args.dataset,
        image_column=args.image_column,
        label_column=args.label_column,
    )

    if args.train_split not in dataset:
        raise ValueError(f"Missing train split {args.train_split!r}. Found: {list(dataset)}")
    eval_split = args.eval_split if args.eval_split in dataset else None

    columns = AffectNetColumns(image=args.image_column, label=args.label_column)
    train_dataset = AffectNetTorchDataset(dataset[args.train_split], processor, columns, label2id)
    eval_dataset = (
        AffectNetTorchDataset(dataset[eval_split], processor, columns, label2id) if eval_split else None
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

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=args.device,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            description=f"epoch {epoch}/{args.epochs} train",
        )

        metrics = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy}
        if eval_loader:
            eval_loss, eval_accuracy = run_epoch(
                model=model,
                loader=eval_loader,
                criterion=criterion,
                device=args.device,
                train=False,
                description=f"epoch {epoch}/{args.epochs} eval",
            )
            metrics.update({"eval_loss": eval_loss, "eval_accuracy": eval_accuracy})

            if eval_accuracy >= best_accuracy:
                best_accuracy = eval_accuracy
                save_checkpoint(args.output_dir, model, processor, label2id, id2label, metrics)
        else:
            save_checkpoint(args.output_dir, model, processor, label2id, id2label, metrics)

        print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    train: bool,
    description: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def save_checkpoint(
    output_dir: Path,
    model: Dinov2EmotionClassifier,
    processor: AutoImageProcessor,
    label2id: dict[str, int],
    id2label: dict[int, str],
    metrics: dict[str, float],
) -> None:
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    processor.save_pretrained(output_dir)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            file,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
