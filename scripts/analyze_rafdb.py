#!/usr/bin/env python3
"""Show RAF-DB label counts and image-size distributions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.face.label_schema import (
    CANONICAL_EMOTION_LABELS,
    split_train_validation_rows,
)
from src.data.face.rafdb_dataset import RAFDB_LABEL_MAP


COMMON_LABEL2ID = {label: index for index, label in enumerate(CANONICAL_EMOTION_LABELS)}
COMMON_LABELS = tuple(COMMON_LABEL2ID)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/raw/face/rafdb")
    parser.add_argument("--image-root", default="DATASET")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--train-labels", default="train_labels.csv")
    parser.add_argument("--test-labels", default="test_labels.csv")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--top-k-sizes", type=int, default=10)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    train_rows = read_labels(dataset_path / args.train_labels)
    train_rows, validation_rows = split_train_validation_rows(
        train_rows,
        image_column=args.image_column,
        label_column=args.label_column,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
    )
    test_rows = read_labels(dataset_path / args.test_labels)
    splits = {
        "train": (train_rows, "train"),
        "validation": (validation_rows, "train"),
        "test": (test_rows, "test"),
    }

    print(f"Dataset: {dataset_path}")
    print_common_labels()
    print(f"Splits: {', '.join(splits)}")
    print()

    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "labels": COMMON_LABEL2ID,
        "source_labels": RAFDB_LABEL_MAP,
        "splits": {},
    }

    for split, (rows, image_split) in splits.items():
        image_index = index_images(dataset_path / args.image_root / image_split)
        split_report = analyze_split(
            split=split,
            rows=rows,
            dataset_path=dataset_path,
            image_root=args.image_root,
            image_split=image_split,
            image_index=image_index,
            image_column=args.image_column,
            label_column=args.label_column,
            top_k_sizes=args.top_k_sizes,
        )
        report["splits"][split] = split_report
        print_split_report(split, split_report, top_k_sizes=args.top_k_sizes)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(f"Wrote JSON report to {args.json_output}")


def read_labels(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find labels file: {labels_path}")

    with labels_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def index_images(split_root: Path) -> dict[str, Path]:
    if not split_root.exists():
        return {}

    image_index: dict[str, Path] = {}
    for path in split_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_index[path.name] = path
    return image_index


def analyze_split(
    split: str,
    rows: list[dict[str, str]],
    dataset_path: Path,
    image_root: str,
    image_split: str,
    image_index: dict[str, Path],
    image_column: str,
    label_column: str,
    top_k_sizes: int,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    size_counts: Counter[tuple[int, int]] = Counter()
    label_size_counts: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    label_widths: dict[str, list[int]] = defaultdict(list)
    label_heights: dict[str, list[int]] = defaultdict(list)
    widths: list[int] = []
    heights: list[int] = []
    missing_images: list[dict[str, str]] = []

    for row in tqdm(rows, desc=f"analyzing {split}"):
        label_id = str(row[label_column])
        label = format_label(label_id)
        image_name = row[image_column]
        image_path = resolve_image_path(
            dataset_path=dataset_path,
            image_root=image_root,
            image_split=image_split,
            label_id=label_id,
            image_name=image_name,
            image_index=image_index,
        )

        label_counts[label] += 1

        if image_path is None:
            missing_images.append({"image": image_name, "label": label_id})
            continue

        with PILImage.open(image_path) as image:
            width, height = image.size

        size_counts[(width, height)] += 1
        label_size_counts[label][(width, height)] += 1
        label_widths[label].append(width)
        label_heights[label].append(height)
        widths.append(width)
        heights.append(height)

    label_summaries = {}
    for label in ordered_labels(label_counts):
        label_summaries[label] = {
            "count": label_counts[label],
            "image_count": len(label_widths[label]),
            "missing_images": label_counts[label] - len(label_widths[label]),
            "image_size_summary": summarize_sizes(label_widths[label], label_heights[label]),
            "top_image_sizes": format_size_counts(label_size_counts[label], top_k_sizes),
        }

    return {
        "total": len(rows),
        "image_count": len(widths),
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "label_counts": {label: label_counts[label] for label in ordered_labels(label_counts)},
        "labels": label_summaries,
        "image_size_summary": summarize_sizes(widths, heights),
        "top_image_sizes": format_size_counts(size_counts, top_k_sizes),
        "top_image_sizes_by_label": {
            label: format_size_counts(label_size_counts[label], top_k_sizes)
            for label in ordered_labels(label_size_counts)
        },
    }


def resolve_image_path(
    dataset_path: Path,
    image_root: str,
    image_split: str,
    label_id: str,
    image_name: str,
    image_index: dict[str, Path],
) -> Path | None:
    image_path = Path(image_name)
    candidates = [
        dataset_path / image_path,
        dataset_path / image_root / image_split / label_id / image_path,
        dataset_path / image_root / image_split / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return image_index.get(image_path.name)


def format_label(label_id: str) -> str:
    return RAFDB_LABEL_MAP.get(label_id, "unknown")


def ordered_labels(counts: Counter[str] | dict[str, object]) -> list[str]:
    known = [label for label in COMMON_LABELS if label in counts]
    remaining = sorted(label for label in counts if label not in COMMON_LABELS)
    return known + remaining


def print_common_labels() -> None:
    labels = sorted(COMMON_LABEL2ID.items(), key=lambda item: item[1])
    print("Labels: " + ", ".join(f"{label_id}:{label}" for label, label_id in labels))


def summarize_sizes(widths: list[int], heights: list[int]) -> dict[str, Any]:
    if not widths or not heights:
        return {}

    return {
        "width_min": min(widths),
        "width_max": max(widths),
        "width_mean": round(sum(widths) / len(widths), 2),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": round(sum(heights) / len(heights), 2),
    }


def format_size_counts(
    size_counts: Counter[tuple[int, int]],
    top_k_sizes: int,
) -> list[dict[str, int]]:
    return [
        {"width": width, "height": height, "count": count}
        for (width, height), count in size_counts.most_common(top_k_sizes)
    ]


def print_split_report(split: str, report: dict[str, Any], top_k_sizes: int) -> None:
    print(
        f"[{split}] total={report['total']}, "
        f"images={report['image_count']}, "
        f"missing_images={report['missing_image_count']}"
    )
    print("Label counts and image sizes")
    print_table(
        ("label", "count", "ratio", "missing", "width min/mean/max", "height min/mean/max", "top size"),
        [
            format_label_row(label, report["labels"][label], report["total"])
            for label in report["labels"]
        ],
    )

    summary = report["image_size_summary"]
    if summary:
        print("Image size summary")
        print(
            "  "
            f"width min/mean/max={summary['width_min']}/{summary['width_mean']}/{summary['width_max']}, "
            f"height min/mean/max={summary['height_min']}/{summary['height_mean']}/{summary['height_max']}"
        )

    print(f"Top {top_k_sizes} image sizes")
    print_table(
        ("size", "count", "ratio"),
        [
            (
                f"{item['width']}x{item['height']}",
                item["count"],
                f"{item['count'] / max(report['image_count'], 1):.2%}",
            )
            for item in report["top_image_sizes"]
        ],
    )
    print()


def format_label_row(label: str, label_report: dict[str, Any], split_total: int) -> tuple[str, Any, str, Any, str, str, str]:
    count = label_report["count"]
    summary = label_report["image_size_summary"]
    top_sizes = label_report["top_image_sizes"]
    top_size = f"{top_sizes[0]['width']}x{top_sizes[0]['height']} ({top_sizes[0]['count']})" if top_sizes else "-"
    width_summary = (
        f"{summary['width_min']}/{summary['width_mean']}/{summary['width_max']}"
        if summary
        else "-"
    )
    height_summary = (
        f"{summary['height_min']}/{summary['height_mean']}/{summary['height_max']}"
        if summary
        else "-"
    )
    return (
        label,
        count,
        f"{count / split_total:.2%}",
        label_report["missing_images"],
        width_summary,
        height_summary,
        top_size,
    )


def print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    table_rows = [headers, *rows]
    widths = [
        max(len(str(row[column])) for row in table_rows)
        for column in range(len(headers))
    ]

    header_line = "  " + "  ".join(
        str(value).ljust(widths[index]) for index, value in enumerate(headers)
    )
    separator = "  " + "  ".join("-" * width for width in widths)
    print(header_line)
    print(separator)
    for row in rows:
        print(
            "  "
            + "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        )


if __name__ == "__main__":
    main()
