#!/usr/bin/env python3
"""Show AffectNet label counts and image-size distributions."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.face.emotion_dataset import load_face_emotion_dataset  # noqa: E402
from src.data.face.label_schema import (  # noqa: E402
    CANONICAL_EMOTION_LABELS,
    canonicalize_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/raw/face/affectnet")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--top-k-sizes", type=int, default=10)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, label2id, _ = load_face_emotion_dataset(
        args.dataset,
        dataset_type="affectnet",
        image_column=args.image_column,
        label_column=args.label_column,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
    )

    print(f"Dataset: {Path(args.dataset).expanduser().resolve()}")
    print_common_labels(label2id)
    print(f"Splits: {', '.join(dataset.keys())}")
    print()

    report: dict[str, Any] = {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "labels": label2id,
        "splits": {},
    }

    for split, split_dataset in dataset.items():
        split_report = analyze_split(
            split_dataset,
            image_column=args.image_column,
            label_column=args.label_column,
            top_k_sizes=args.top_k_sizes,
            description=f"analyzing {split}",
        )
        report["splits"][split] = split_report
        print_split_report(split, split_report, top_k_sizes=args.top_k_sizes)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(f"Wrote JSON report to {args.json_output}")


def analyze_split(
    split_dataset: Any,
    image_column: str,
    label_column: str,
    top_k_sizes: int,
    description: str,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    size_counts: Counter[tuple[int, int]] = Counter()
    label_size_counts: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    label_widths: dict[str, list[int]] = defaultdict(list)
    label_heights: dict[str, list[int]] = defaultdict(list)
    widths: list[int] = []
    heights: list[int] = []

    for row in tqdm(split_dataset, desc=description):
        label = canonicalize_label(str(row[label_column]))
        width, height = read_image_size(row, image_column=image_column)

        label_counts[label] += 1
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
            "image_size_summary": summarize_sizes(label_widths[label], label_heights[label]),
            "top_image_sizes": format_size_counts(label_size_counts[label], top_k_sizes),
        }

    return {
        "total": sum(label_counts.values()),
        "label_counts": {label: label_counts[label] for label in ordered_labels(label_counts)},
        "labels": label_summaries,
        "image_size_summary": summarize_sizes(widths, heights),
        "top_image_sizes": format_size_counts(size_counts, top_k_sizes),
        "top_image_sizes_by_label": {
            label: format_size_counts(label_size_counts[label], top_k_sizes)
            for label in ordered_labels(label_size_counts)
        },
    }


def ordered_labels(counts: Counter[str] | dict[str, object]) -> list[str]:
    known = [label for label in CANONICAL_EMOTION_LABELS if label in counts]
    remaining = sorted(label for label in counts if label not in CANONICAL_EMOTION_LABELS)
    return known + remaining


def print_common_labels(label2id: dict[str, int]) -> None:
    labels = sorted(label2id.items(), key=lambda item: item[1])
    print("Labels: " + ", ".join(f"{label_id}:{label}" for label, label_id in labels))


def read_image_size(row: dict[str, Any], image_column: str) -> tuple[int, int]:
    image_value = row.get(image_column)

    if isinstance(image_value, PILImage.Image):
        return image_value.size

    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            with PILImage.open(io.BytesIO(image_value["bytes"])) as image:
                return image.size
        if image_value.get("path"):
            with PILImage.open(image_value["path"]) as image:
                return image.size

    if isinstance(image_value, (str, Path)):
        image_path = Path(image_value)
        if image_path.exists():
            with PILImage.open(image_path) as image:
                return image.size

        zip_path = row.get("__affectnet_zip__")
        if zip_path:
            with zipfile.ZipFile(zip_path) as zip_file:
                with zip_file.open(str(image_value).replace("\\", "/")) as file:
                    with PILImage.open(io.BytesIO(file.read())) as image:
                        return image.size

    raise FileNotFoundError(f"Could not read image size from row keys: {sorted(row)}")


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
    print(f"[{split}] total={report['total']}")
    print("Label counts and image sizes")
    print_table(
        ("label", "count", "ratio", "width min/mean/max", "height min/mean/max", "top size"),
        [
            format_label_row(label, report["labels"][label], report["total"])
            for label in report["labels"]
        ],
    )

    summary = report["image_size_summary"]
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
                f"{item['count'] / report['total']:.2%}",
            )
            for item in report["top_image_sizes"]
        ],
    )
    print()


def format_label_row(label: str, label_report: dict[str, Any], split_total: int) -> tuple[str, Any, str, str, str, str]:
    count = label_report["count"]
    summary = label_report["image_size_summary"]
    top_size = label_report["top_image_sizes"][0]
    return (
        label,
        count,
        f"{count / split_total:.2%}",
        f"{summary['width_min']}/{summary['width_mean']}/{summary['width_max']}",
        f"{summary['height_min']}/{summary['height_mean']}/{summary['height_max']}",
        f"{top_size['width']}x{top_size['height']} ({top_size['count']})",
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
