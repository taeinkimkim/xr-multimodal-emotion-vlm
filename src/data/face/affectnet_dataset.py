"""AffectNet dataset loader preset.

This module keeps the older AffectNet-specific API available while delegating
to the shared face-emotion training loader.
"""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict

from src.data.face.generic_dataset import load_generic_dataset
from src.data.face.label_schema import CANONICAL_EMOTION_LABELS, label_mappings


CANONICAL_AFFECTNET_LABELS = CANONICAL_EMOTION_LABELS
# The training heads in this project use the shared 7-class face-emotion schema.
# Folder datasets such as balanced_affectnet may include Contempt, which is
# intentionally excluded here to keep labels aligned with CANONICAL_EMOTION_LABELS.
EXCLUDED_AFFECTNET_LABELS = {"contempt"}
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42


def load_affectnet_dataset(
    dataset_path: str | Path,
    image_column: str = "image",
    label_column: str = "label",
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """Load AffectNet with the shared 7-class face-emotion label schema.

    Pre-split folder datasets, for example train/val/test class folders, are
    handled by the generic loader and keep their existing validation split.
    """

    dataset = load_generic_dataset(
        dataset_path,
        image_column=image_column,
        label_column=label_column,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        exclude_labels=EXCLUDED_AFFECTNET_LABELS,
    )
    label2id, id2label = label_mappings()
    return dataset, label2id, id2label
