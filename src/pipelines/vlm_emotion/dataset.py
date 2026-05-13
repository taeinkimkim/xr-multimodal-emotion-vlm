"""Test dataset loader for VLM emotion experiments."""

from __future__ import annotations

from pathlib import Path

from src.data.face.label_schema import canonicalize_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_test_samples(test_dir: Path | str) -> list[dict[str, str]]:
    """Scan test_dir/{emotion}/{image} folder and return list of {image_path, label} dicts."""
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    samples: list[dict[str, str]] = []
    for label_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        if label_dir.name.startswith("."):
            continue
        label = canonicalize_label(label_dir.name)
        for image_path in sorted(
            p for p in label_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ):
            samples.append({"image_path": str(image_path), "label": label})

    if not samples:
        raise ValueError(f"No images found under {test_dir}")
    return samples
