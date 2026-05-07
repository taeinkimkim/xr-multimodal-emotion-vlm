"""Output file, checkpoint, and logging utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

import torch
from torch import nn


def make_output_filenames(model_filename: str) -> dict[str, str]:
    model_path = Path(model_filename)
    if model_path.name != model_filename:
        raise ValueError("--model-filename must be a filename, not a path")

    suffix = model_path.suffix or ".pt"
    return {
        "model": f"model{suffix}",
        "metrics": "metrics.json",
        "log": "log.txt",
        "history": "history.json",
        "best_model": f"best{suffix}",
        "best_metrics": "best_metrics.json",
    }


def resolve_output_dir(base_output_dir: Path, model_filename: str) -> Path:
    model_path = Path(model_filename)
    if model_path.name != model_filename:
        raise ValueError("--model-filename must be a filename, not a path")

    stem = model_path.stem
    if not stem:
        raise ValueError("--model-filename must include a non-empty filename")
    return base_output_dir / stem


def save_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    _processor: Any,
    label2id: dict[str, int],
    id2label: dict[int, str],
    metrics: dict[str, object],
    model_filename: str = "pytorch_model.bin",
    metrics_filename: str = "metrics.json",
) -> None:
    torch.save(model.state_dict(), output_dir / model_filename)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            file,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / metrics_filename).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


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
