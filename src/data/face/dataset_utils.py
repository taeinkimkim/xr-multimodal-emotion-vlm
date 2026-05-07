"""Reusable dataset-loading utilities for face-emotion datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src.data.face.label_schema import (
    CANONICAL_EMOTION_LABEL_SET,
    canonicalize_label,
    split_train_validation_rows,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset_dict(
    path: Path,
    image_column: str,
    label_column: str,
    validation_ratio: float,
    split_seed: int,
    exclude_labels: set[str] | None = None,
) -> DatasetDict:
    exclude_labels = exclude_labels or set()

    if path.is_dir() and (path / "dataset_info.json").exists():
        loaded = load_from_disk(str(path))
        return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})

    if path.is_dir():
        folder_dataset = load_folder_splits(
            path,
            image_column=image_column,
            label_column=label_column,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
            exclude_labels=exclude_labels,
        )
        if folder_dataset is not None:
            return folder_dataset

    data_files = discover_data_files(path)
    if data_files:
        extension = dataset_extension(next(iter(data_files.values())))
        loaded = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=str(cache_dir_for(path)),
        )
        return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})

    if path.is_dir():
        imagefolder = try_imagefolder(path)
        if imagefolder is not None:
            return imagefolder

    raise FileNotFoundError(
        f"Could not find a loadable face-emotion dataset at {path}. "
        "Expected save_to_disk files, labels.csv, split csv/json/parquet files, or image folders."
    )


def discover_data_files(path: Path) -> dict[str, str]:
    if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}:
        return {"train": str(path)}

    if not path.is_dir():
        return {}

    split_names = {
        "train": ("train", "training"),
        "validation": ("validation", "valid", "val", "dev"),
        "test": ("test", "testing"),
    }
    files: dict[str, str] = {}
    for split, candidates in split_names.items():
        for name in candidates:
            for suffix in (".parquet", ".csv", ".jsonl", ".json"):
                candidate = path / f"{name}{suffix}"
                if candidate.exists():
                    files[split] = str(candidate)
                    break
            if split in files:
                break

    if not files:
        for name in ("labels.csv", "metadata.csv"):
            candidate = path / name
            if candidate.exists():
                files["train"] = str(candidate)
                break

    return files


def dataset_extension(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".jsonl":
        return "json"
    return suffix.removeprefix(".")


def try_imagefolder(path: Path) -> DatasetDict | None:
    if not any(file.suffix.lower() in IMAGE_SUFFIXES for file in path.rglob("*")):
        return None
    loaded = load_dataset(
        "imagefolder",
        data_dir=str(path),
        cache_dir=str(cache_dir_for(path)),
    )
    return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})


def load_folder_splits(
    path: Path,
    image_column: str,
    label_column: str,
    validation_ratio: float,
    split_seed: int,
    exclude_labels: set[str],
) -> DatasetDict | None:
    train_dir = first_existing_dir(path, "Train", "train", "Training", "training")
    validation_dir = first_existing_dir(path, "Validation", "validation", "Valid", "valid", "Val", "val")
    test_dir = first_existing_dir(path, "Test", "test", "Testing", "testing")

    if train_dir is None and looks_like_label_folder(path):
        train_dir = path

    if train_dir is None and validation_dir is None and test_dir is None:
        return None

    dataset = DatasetDict()
    if train_dir is not None:
        train_rows = scan_label_folder(train_dir, image_column, label_column, exclude_labels)
        if validation_dir is None:
            train_rows, validation_rows = split_train_validation_rows(
                train_rows,
                image_column=image_column,
                label_column=label_column,
                validation_ratio=validation_ratio,
                split_seed=split_seed,
            )
        else:
            validation_rows = scan_label_folder(validation_dir, image_column, label_column, exclude_labels)

        if train_rows:
            dataset["train"] = Dataset.from_list(train_rows)
        if validation_rows:
            dataset["validation"] = Dataset.from_list(validation_rows)

    if train_dir is None and validation_dir is not None:
        validation_rows = scan_label_folder(validation_dir, image_column, label_column, exclude_labels)
        if validation_rows:
            dataset["validation"] = Dataset.from_list(validation_rows)

    if test_dir is not None:
        test_rows = scan_label_folder(test_dir, image_column, label_column, exclude_labels)
        if test_rows:
            dataset["test"] = Dataset.from_list(test_rows)

    return dataset if dataset else None


def first_existing_dir(root: Path, *names: str) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def looks_like_label_folder(path: Path) -> bool:
    return any(child.is_dir() for child in path.iterdir()) and any(
        file.suffix.lower() in IMAGE_SUFFIXES for file in path.rglob("*")
    )


def scan_label_folder(
    split_dir: Path,
    image_column: str,
    label_column: str,
    exclude_labels: set[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
        label = canonicalize_label(label_dir.name)
        if label.lower() in exclude_labels or label in exclude_labels:
            continue
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({image_column: str(image_path.resolve()), label_column: label})
    return rows


def normalize_label_values(dataset: DatasetDict, label_column: str) -> DatasetDict:
    normalized = DatasetDict()
    for split, split_dataset in dataset.items():
        if label_column not in split_dataset.column_names:
            normalized[split] = split_dataset
            continue
        label_feature = split_dataset.features.get(label_column)
        label_names = getattr(label_feature, "names", None)

        def normalize_row(row: dict[str, Any]) -> dict[str, str]:
            raw_label = row[label_column]
            if label_names and isinstance(raw_label, int):
                raw_label = label_names[raw_label]
            return {label_column: canonicalize_label(str(raw_label))}

        normalized[split] = split_dataset.map(normalize_row)
    return normalized


def filter_supported_labels(dataset: DatasetDict, label_column: str) -> DatasetDict:
    filtered = DatasetDict()
    for split, split_dataset in dataset.items():
        if label_column not in split_dataset.column_names:
            filtered[split] = split_dataset
            continue
        filtered[split] = split_dataset.filter(
            lambda row: row[label_column] in CANONICAL_EMOTION_LABEL_SET
        )
    return filtered


def normalize_local_columns(
    dataset: DatasetDict,
    dataset_path: Path,
    image_column: str,
) -> DatasetDict:
    if not dataset_path.exists():
        return dataset

    root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    zip_path = find_zip(root)
    normalized = DatasetDict()

    for split, split_dataset in dataset.items():
        columns = set(split_dataset.column_names)
        source_column = next(
            (column for column in ("image", "img", "path", "pth", "file", "filename") if column in columns),
            None,
        )
        if image_column not in columns and source_column is None:
            normalized[split] = split_dataset
            continue

        if image_column not in columns:
            split_dataset = split_dataset.add_column(image_column, split_dataset[source_column])
        columns = set(split_dataset.column_names)

        if "__face_emotion_root__" not in columns:
            split_dataset = split_dataset.add_column(
                "__face_emotion_root__", [str(root)] * len(split_dataset)
            )
        if zip_path and "__face_emotion_zip__" not in columns:
            split_dataset = split_dataset.add_column(
                "__face_emotion_zip__", [str(zip_path)] * len(split_dataset)
            )
        normalized[split] = split_dataset

    return normalized


def cache_dir_for(path: Path) -> Path:
    root = path if path.is_dir() else path.parent
    return root / ".cache" / "hf_datasets"


def find_zip(root: Path) -> Path | None:
    preferred = root / "affectnet.zip"
    if preferred.exists():
        return preferred
    zip_files = sorted(root.glob("*.zip"))
    return zip_files[0] if zip_files else None
