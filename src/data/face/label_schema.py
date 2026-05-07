"""Shared face-emotion label schema and split helpers."""

from __future__ import annotations

import random
from collections import defaultdict


CANONICAL_EMOTION_LABELS = (
    "Happiness",
    "Sadness",
    "Anger",
    "Surprise",
    "Fear",
    "Disgust",
    "Neutral",
)
CANONICAL_EMOTION_LABEL_SET = set(CANONICAL_EMOTION_LABELS)
LABEL_ALIASES = {
    "happiness": "Happiness",
    "happy": "Happiness",
    "sadness": "Sadness",
    "sad": "Sadness",
    "anger": "Anger",
    "angry": "Anger",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "fear": "Fear",
    "fearful": "Fear",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "neutral": "Neutral",
}


def canonicalize_label(label: str) -> str:
    normalized = label.strip()
    return LABEL_ALIASES.get(normalized.lower(), normalized)


def label_mappings() -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: index for index, label in enumerate(CANONICAL_EMOTION_LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def split_train_validation_rows(
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
        grouped_rows[str(row[label_column])].append(row)

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
