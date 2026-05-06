"""AffectNet dataset helpers for DINOv2 training.

The loader supports these common layouts:
- A local folder dataset such as ``Train/happy/*.jpg`` and ``Test/Anger/*.png``.
- Hugging Face datasets saved with ``save_to_disk`` or snapshot files.
- A local ``labels.csv`` with image paths in ``pth`` and images stored either
  extracted on disk or inside ``affectnet.zip``.
"""

from __future__ import annotations

import io
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image as PILImage
from torch.utils.data import Dataset as TorchDataset


CANONICAL_AFFECTNET_LABELS = (
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
)
EXCLUDED_AFFECTNET_LABELS = {"contempt"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True)
class AffectNetColumns:
    """Column names used by the training pipeline."""

    image: str = "image"
    label: str = "label"


def load_affectnet_dataset(
    dataset_path: str | Path,
    image_column: str = "image",
    label_column: str = "label",
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """Load AffectNet data and infer label mappings.

    Args:
        dataset_path: Local dataset directory or file.
        image_column: Preferred image column name.
        label_column: Label column name.
        validation_ratio: Fraction of folder-based Train data reserved for validation.
        split_seed: Random seed used for the Train/validation split.

    Returns:
        A ``DatasetDict`` plus ``label2id`` and ``id2label`` mappings.
    """

    path = Path(dataset_path).expanduser().resolve()
    dataset = _load_dataset_dict(
        path,
        image_column=image_column,
        label_column=label_column,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
    )
    dataset = _normalize_local_columns(dataset, path, image_column=image_column)
    label2id, id2label = _build_label_mappings(dataset, label_column)
    return dataset, label2id, id2label


class AffectNetTorchDataset(TorchDataset):
    """Torch wrapper that applies a Hugging Face image processor."""

    def __init__(
        self,
        dataset: Dataset,
        processor: Any,
        columns: AffectNetColumns,
        label2id: dict[str, int],
    ) -> None:
        self.dataset = dataset
        self.processor = processor
        self.columns = columns
        self.label2id = label2id
        self._zip_files: dict[str, zipfile.ZipFile] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataset[index]
        image = self._load_image(row).convert("RGB")
        label = self._label_to_id(row[self.columns.label])

        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _label_to_id(self, label: Any) -> int:
        if isinstance(label, int):
            return label
        return self.label2id[str(label)]

    def _load_image(self, row: dict[str, Any]) -> PILImage.Image:
        image_value = _first_present(row, self.columns.image, "image", "img", "path", "pth", "file")

        if isinstance(image_value, PILImage.Image):
            return image_value

        if isinstance(image_value, dict):
            if image_value.get("bytes") is not None:
                return PILImage.open(io.BytesIO(image_value["bytes"]))
            if image_value.get("path"):
                return PILImage.open(image_value["path"])

        if isinstance(image_value, (str, Path)):
            image_path = Path(image_value)
            if image_path.exists():
                return PILImage.open(image_path)

            root = row.get("__affectnet_root__")
            if root:
                rooted_path = Path(root) / image_path
                if rooted_path.exists():
                    return PILImage.open(rooted_path)

            zip_path = row.get("__affectnet_zip__")
            if zip_path:
                return self._load_from_zip(str(zip_path), str(image_value))

        raise FileNotFoundError(f"Could not load image from row keys: {sorted(row)}")

    def _load_from_zip(self, zip_path: str, member_path: str) -> PILImage.Image:
        zip_file = self._zip_files.get(zip_path)
        if zip_file is None:
            zip_file = zipfile.ZipFile(zip_path)
            self._zip_files[zip_path] = zip_file

        normalized = member_path.replace("\\", "/")
        with zip_file.open(normalized) as file:
            return PILImage.open(io.BytesIO(file.read()))


def _load_dataset_dict(
    path: Path,
    image_column: str,
    label_column: str,
    validation_ratio: float,
    split_seed: int,
) -> DatasetDict:
    if path.is_dir() and (path / "dataset_info.json").exists():
        loaded = load_from_disk(str(path))
        return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})

    if path.is_dir():
        folder_dataset = _load_folder_splits(
            path,
            image_column=image_column,
            label_column=label_column,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
        )
        if folder_dataset is not None:
            return folder_dataset

    data_files = _discover_data_files(path)
    if data_files:
        extension = _dataset_extension(next(iter(data_files.values())))
        loaded = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=str(_cache_dir_for(path)),
        )
        return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})

    if path.is_dir():
        imagefolder = _try_imagefolder(path)
        if imagefolder is not None:
            return imagefolder

    raise FileNotFoundError(
        f"Could not find a loadable AffectNet dataset at {path}. "
        "Expected save_to_disk files, labels.csv, split csv/json/parquet files, or image folders."
    )


def _discover_data_files(path: Path) -> dict[str, str]:
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


def _dataset_extension(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".jsonl":
        return "json"
    return suffix.removeprefix(".")


def _try_imagefolder(path: Path) -> DatasetDict | None:
    if not any(file.suffix.lower() in IMAGE_SUFFIXES for file in path.rglob("*")):
        return None
    loaded = load_dataset(
        "imagefolder",
        data_dir=str(path),
        cache_dir=str(_cache_dir_for(path)),
    )
    return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})


def _load_folder_splits(
    path: Path,
    image_column: str,
    label_column: str,
    validation_ratio: float,
    split_seed: int,
) -> DatasetDict | None:
    train_dir = _first_existing_dir(path, "Train", "train", "Training", "training")
    validation_dir = _first_existing_dir(path, "Validation", "validation", "Valid", "valid", "Val", "val")
    test_dir = _first_existing_dir(path, "Test", "test", "Testing", "testing")

    if train_dir is None and _looks_like_label_folder(path):
        train_dir = path

    if train_dir is None and validation_dir is None and test_dir is None:
        return None

    dataset = DatasetDict()

    if train_dir is not None:
        train_rows = _scan_label_folder(train_dir, image_column=image_column, label_column=label_column)
        if validation_dir is None:
            train_rows, validation_rows = _split_train_validation(
                train_rows,
                image_column=image_column,
                label_column=label_column,
                validation_ratio=validation_ratio,
                split_seed=split_seed,
            )
        else:
            validation_rows = _scan_label_folder(
                validation_dir,
                image_column=image_column,
                label_column=label_column,
            )

        if train_rows:
            dataset["train"] = Dataset.from_list(train_rows)
        if validation_rows:
            dataset["validation"] = Dataset.from_list(validation_rows)

    if train_dir is None and validation_dir is not None:
        validation_rows = _scan_label_folder(
            validation_dir,
            image_column=image_column,
            label_column=label_column,
        )
        if validation_rows:
            dataset["validation"] = Dataset.from_list(validation_rows)

    if test_dir is not None:
        test_rows = _scan_label_folder(test_dir, image_column=image_column, label_column=label_column)
        if test_rows:
            dataset["test"] = Dataset.from_list(test_rows)

    return dataset if dataset else None


def _split_train_validation(
    rows: list[dict[str, str]],
    image_column: str,
    label_column: str,
    validation_ratio: float,
    split_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < validation_ratio < 1.0:
        return rows, []

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row[label_column]].append(row)

    rng = random.Random(split_seed)
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []

    for label in sorted(grouped_rows):
        label_rows = grouped_rows[label]
        rng.shuffle(label_rows)

        validation_count = round(len(label_rows) * validation_ratio)
        if len(label_rows) > 1:
            validation_count = max(1, min(validation_count, len(label_rows) - 1))
        else:
            validation_count = 0

        validation_rows.extend(label_rows[:validation_count])
        train_rows.extend(label_rows[validation_count:])

    train_rows.sort(key=lambda row: row.get(image_column, ""))
    validation_rows.sort(key=lambda row: row.get(image_column, ""))
    return train_rows, validation_rows


def _first_existing_dir(root: Path, *names: str) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _looks_like_label_folder(path: Path) -> bool:
    return any(child.is_dir() for child in path.iterdir()) and any(
        file.suffix.lower() in IMAGE_SUFFIXES for file in path.rglob("*")
    )


def _scan_label_folder(
    split_dir: Path,
    image_column: str,
    label_column: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
        label = _normalize_label(label_dir.name)
        if label in EXCLUDED_AFFECTNET_LABELS:
            continue
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({image_column: str(image_path.resolve()), label_column: label})
    return rows


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _cache_dir_for(path: Path) -> Path:
    root = path if path.is_dir() else path.parent
    return root / ".cache" / "hf_datasets"


def _normalize_local_columns(
    dataset: DatasetDict,
    dataset_path: Path,
    image_column: str,
) -> DatasetDict:
    if not dataset_path.exists():
        return dataset

    root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    zip_path = _find_zip(root)
    normalized = DatasetDict()

    for split, split_dataset in dataset.items():
        columns = set(split_dataset.column_names)
        source_column = next((column for column in ("image", "img", "path", "pth", "file") if column in columns), None)
        if image_column not in columns and source_column is None:
            normalized[split] = split_dataset
            continue

        if image_column not in columns:
            split_dataset = split_dataset.add_column(image_column, split_dataset[source_column])
        columns = set(split_dataset.column_names)

        if "__affectnet_root__" not in columns:
            split_dataset = split_dataset.add_column(
                "__affectnet_root__", [str(root)] * len(split_dataset)
            )
        if zip_path and "__affectnet_zip__" not in columns:
            split_dataset = split_dataset.add_column(
                "__affectnet_zip__", [str(zip_path)] * len(split_dataset)
            )
        normalized[split] = split_dataset

    return normalized


def _find_zip(root: Path) -> Path | None:
    preferred = root / "affectnet.zip"
    if preferred.exists():
        return preferred
    zip_files = sorted(root.glob("*.zip"))
    return zip_files[0] if zip_files else None


def _build_label_mappings(
    dataset: DatasetDict,
    label_column: str,
) -> tuple[dict[str, int], dict[int, str]]:
    first_split = next(iter(dataset.values()))
    label_feature = first_split.features.get(label_column)
    names = getattr(label_feature, "names", None)

    if names:
        id2label = {index: str(label) for index, label in enumerate(names)}
        label2id = {label: index for index, label in id2label.items()}
        return label2id, id2label

    labels: list[str] = []
    seen: set[str] = set()
    for split_dataset in dataset.values():
        for label in split_dataset[label_column]:
            label_name = str(label)
            if label_name not in seen:
                labels.append(label_name)
                seen.add(label_name)

    labels = _ordered_labels(labels)
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def _ordered_labels(labels: list[str]) -> list[str]:
    canonical = [label for label in CANONICAL_AFFECTNET_LABELS if label in labels]
    remaining = sorted(label for label in labels if label not in canonical)
    return canonical + remaining


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None
