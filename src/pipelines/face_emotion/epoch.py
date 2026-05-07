"""Training and evaluation loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pipelines.face_emotion.metrics import (
    finalize_evaluation_metrics,
    new_evaluation_stats,
    update_class_stats,
)


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
    class_stats = new_evaluation_stats(len(id2label or {}), ece_bins) if collect_class_metrics else None

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
            update_class_stats(
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
            finalize_evaluation_metrics(
                class_stats,
                id2label,
                top_confused_pairs=top_confused_pairs,
            )
        )
    return metrics
