"""Generic face-emotion dataset helpers for DINOv2 training."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from PIL import Image as PILImage
from torch.utils.data import Dataset as TorchDataset

from src.data.face.affectnet_dataset import load_affectnet_dataset
from src.data.face.generic_dataset import load_generic_dataset
from src.data.face.label_schema import canonicalize_label, label_mappings
from src.data.face.rafdb_dataset import RAFDB_LABEL_MAP, load_rafdb_dataset


DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True)
class FaceEmotionColumns:
    """Column names used by the training pipeline."""

    image: str = "image"
    label: str = "label"


def load_face_emotion_dataset(
    dataset_path: str | Path,
    dataset_type: str = "auto",
    image_column: str = "image",
    label_column: str = "label",
    image_root: str = "DATASET",
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
    label_map: dict[str, str] | None = None,
) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """Load a face-emotion dataset and infer label mappings."""

    path = Path(dataset_path).expanduser().resolve()
    normalized_type = dataset_type.lower()
    if normalized_type not in {"auto", "affectnet", "rafdb", "generic"}:
        raise ValueError("--dataset-type must be one of: auto, affectnet, rafdb, generic")

    if normalized_type == "auto":
        normalized_type = _detect_dataset_type(path)

    if normalized_type == "affectnet":
        dataset, label2id, id2label = load_affectnet_dataset(
            path,
            image_column=image_column,
            label_column=label_column,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
        )
        return dataset, label2id, id2label

    if normalized_type == "rafdb":
        dataset = load_rafdb_dataset(
            path,
            image_column=image_column,
            label_column=label_column,
            image_root=image_root,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
            label_map=label_map or RAFDB_LABEL_MAP,
        )
    else:
        dataset = load_generic_dataset(
            path,
            image_column=image_column,
            label_column=label_column,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
        )

    label2id, id2label = label_mappings()
    return dataset, label2id, id2label


class FaceEmotionTorchDataset(TorchDataset):
    """Torch wrapper that applies a Hugging Face image processor."""

    def __init__(
        self,
        dataset: Dataset,
        processor: Any,
        columns: FaceEmotionColumns,
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
            label_as_text = str(label)
            if label_as_text in self.label2id:
                return self.label2id[label_as_text]
            return label
        return self.label2id[str(label)]

    def _load_image(self, row: dict[str, Any]) -> PILImage.Image:
        image_value = _first_present(
            row,
            self.columns.image,
            "image",
            "img",
            "path",
            "pth",
            "file",
            "filename",
        )

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

            root = _first_present(row, "__face_emotion_root__", "__affectnet_root__")
            if root:
                rooted_path = Path(root) / image_path
                if rooted_path.exists():
                    return PILImage.open(rooted_path)

            zip_path = _first_present(row, "__face_emotion_zip__", "__affectnet_zip__")
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


def parse_label_map(value: str | None) -> dict[str, str] | None:
    if not value:
        return None

    path = Path(value).expanduser()
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    else:
        loaded = json.loads(value)

    if not isinstance(loaded, dict):
        raise ValueError("--label-map must be a JSON object or a path to one")
    return {str(key): canonicalize_label(str(label)) for key, label in loaded.items()}


def _detect_dataset_type(path: Path) -> str:
    if "rafdb" in path.name.lower():
        return "rafdb"
    if "affectnet" in path.name.lower():
        return "affectnet"
    return "generic"


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None
