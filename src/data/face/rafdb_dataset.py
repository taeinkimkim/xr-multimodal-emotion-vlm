"""RAF-DB dataset loader."""

from __future__ import annotations

import csv
from pathlib import Path

from datasets import Dataset, DatasetDict

from src.data.face.dataset_utils import filter_supported_labels
from src.data.face.label_schema import canonicalize_label, split_train_validation_rows


RAFDB_LABEL_MAP = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rafdb_dataset(
    dataset_path: str | Path,
    image_column: str = "image",
    label_column: str = "label",
    image_root: str = "DATASET",
    validation_ratio: float = 0.1,
    split_seed: int = 42,
    label_map: dict[str, str] | None = None,
) -> DatasetDict:
    path = Path(dataset_path).expanduser().resolve()
    active_label_map = label_map or RAFDB_LABEL_MAP

    train_rows = _read_csv_rows(path / "train_labels.csv")
    test_rows = _read_csv_rows(path / "test_labels.csv")
    train_rows, validation_rows = split_train_validation_rows(
        train_rows,
        image_column=image_column,
        label_column=label_column,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                _normalize_rows(
                    train_rows,
                    path,
                    image_root,
                    "train",
                    image_column,
                    label_column,
                    active_label_map,
                )
            ),
            "validation": Dataset.from_list(
                _normalize_rows(
                    validation_rows,
                    path,
                    image_root,
                    "train",
                    image_column,
                    label_column,
                    active_label_map,
                )
            ),
            "test": Dataset.from_list(
                _normalize_rows(
                    test_rows,
                    path,
                    image_root,
                    "test",
                    image_column,
                    label_column,
                    active_label_map,
                )
            ),
        }
    )
    return filter_supported_labels(dataset, label_column=label_column)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {key: str(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def _normalize_rows(
    rows: list[dict[str, str]],
    dataset_path: Path,
    image_root: str,
    split: str,
    image_column: str,
    label_column: str,
    label_map: dict[str, str],
) -> list[dict[str, str]]:
    normalized_rows: list[dict[str, str]] = []
    image_index = _index_images(dataset_path / image_root / split)
    for row in rows:
        label_id = str(row[label_column])
        label = canonicalize_label(label_map.get(label_id, label_id))
        image_name = str(row[image_column])
        image_path = _resolve_image_path(
            dataset_path=dataset_path,
            image_root=image_root,
            split=split,
            label_id=label_id,
            image_name=image_name,
            image_index=image_index,
        )
        if image_path is None:
            image_path = dataset_path / image_name
        normalized_rows.append({image_column: str(image_path), label_column: label})
    return normalized_rows


def _resolve_image_path(
    dataset_path: Path,
    image_root: str,
    split: str,
    label_id: str,
    image_name: str,
    image_index: dict[str, Path],
) -> Path | None:
    image_path = Path(image_name)
    candidates = [
        dataset_path / image_path,
        dataset_path / image_root / split / label_id / image_path,
        dataset_path / image_root / split / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return image_index.get(image_path.name)


def _index_images(split_root: Path) -> dict[str, Path]:
    if not split_root.exists():
        return {}
    return {
        path.name: path
        for path in split_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
