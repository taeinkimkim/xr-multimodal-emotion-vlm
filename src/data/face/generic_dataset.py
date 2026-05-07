"""Generic face-emotion dataset loader."""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict

from src.data.face.dataset_utils import (
    filter_supported_labels,
    load_dataset_dict,
    normalize_label_values,
    normalize_local_columns,
)


def load_generic_dataset(
    dataset_path: str | Path,
    image_column: str = "image",
    label_column: str = "label",
    validation_ratio: float = 0.1,
    split_seed: int = 42,
    exclude_labels: set[str] | None = None,
) -> DatasetDict:
    path = Path(dataset_path).expanduser().resolve()
    dataset = load_dataset_dict(
        path,
        image_column=image_column,
        label_column=label_column,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        exclude_labels=exclude_labels or set(),
    )
    dataset = normalize_local_columns(dataset, path, image_column=image_column)
    dataset = normalize_label_values(dataset, label_column=label_column)
    return filter_supported_labels(dataset, label_column=label_column)
