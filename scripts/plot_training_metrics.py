#!/usr/bin/env python3
"""Plot training metrics saved by train_face_dinov2.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_json", type=Path, help="Path to *_history.json")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--class-output",
        type=Path,
        default=None,
        help="Output PNG path for per-class metric trends",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_history_report(args.history_json)
    output = args.output or args.history_json.with_name(
        args.history_json.name.replace("_history.json", "_plots.png")
    )
    class_output = args.class_output or args.history_json.with_name(
        args.history_json.name.replace("_history.json", "_class_metrics.png")
    )
    save_training_plots(
        history=report["epochs"],
        final_metrics=report.get("final", {}),
        plot_path=output,
    )
    save_class_metric_plots(
        history=report["epochs"],
        plot_path=class_output,
    )


def load_history_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "epochs" in data:
        return data

    epoch = data.get("epoch")
    if epoch is not None:
        return {"epochs": [data], "final": data}

    raise ValueError(f"{path} does not look like a training history or metrics JSON file")


def save_training_plots(
    history: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    plot_path: Path,
) -> None:
    if not history:
        raise ValueError("No epoch history found to plot")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = plot_path.parent / ".matplotlib"
    cache_dir = plot_path.parent / ".cache"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install matplotlib to generate training plots") from error

    epochs = [int(item["epoch"]) for item in history]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DINOv2 AffectNet Training Summary", fontsize=16)

    plot_metric_lines(
        axes[0, 0],
        epochs,
        history,
        [
            ("train_loss", "train loss"),
            ("eval_loss", "validation loss"),
        ],
        title="Loss",
        ylabel="loss",
    )
    plot_metric_lines(
        axes[0, 1],
        epochs,
        history,
        [
            ("train_accuracy", "train accuracy"),
            ("eval_accuracy", "validation accuracy"),
            ("eval_balanced_accuracy", "balanced accuracy"),
        ],
        title="Accuracy",
        ylabel="score",
    )
    plot_metric_lines(
        axes[0, 2],
        epochs,
        history,
        [
            ("eval_macro_f1", "macro F1"),
            ("eval_weighted_f1", "weighted F1"),
        ],
        title="F1",
        ylabel="score",
    )
    plot_metric_lines(
        axes[1, 0],
        epochs,
        history,
        [
            ("eval_mean_confidence", "mean confidence"),
            ("eval_correct_confidence", "correct confidence"),
            ("eval_incorrect_confidence", "incorrect confidence"),
        ],
        title="Confidence",
        ylabel="softmax",
    )
    plot_metric_lines(
        axes[1, 1],
        epochs,
        history,
        [("eval_ece", "ECE")],
        title="Calibration Error",
        ylabel="ECE",
    )
    plot_confusion_matrix(
        axes[1, 2],
        pick_confusion_matrix(final_metrics, history),
        title="Final Confusion Matrix",
    )

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved training plots to {plot_path}")


def plot_metric_lines(
    axis: Any,
    epochs: list[int],
    history: list[dict[str, Any]],
    series: list[tuple[str, str]],
    title: str,
    ylabel: str,
) -> None:
    plotted = False
    for key, label in series:
        values = [item.get(key) for item in history]
        if all(value is None for value in values):
            continue
        axis.plot(
            epochs,
            [float(value) if value is not None else float("nan") for value in values],
            marker="o",
            label=label,
        )
        plotted = True
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.set_xticks(epochs)
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)


def save_class_metric_plots(
    history: list[dict[str, Any]],
    plot_path: Path,
) -> None:
    labels = collect_class_labels(history)
    if not labels:
        print("No eval_per_class_metrics found; skipped per-class plot.")
        return

    configure_matplotlib_cache(plot_path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install matplotlib to generate training plots") from error

    epochs = [int(item["epoch"]) for item in history]
    metrics_to_plot = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("avg_true_class_softmax", "True-Class Softmax"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Per-Class Validation Metrics by Epoch", fontsize=16)

    for axis, (metric_name, title) in zip(axes.flat, metrics_to_plot):
        for label in labels:
            values = [
                get_class_metric(epoch_metrics, label, metric_name)
                for epoch_metrics in history
            ]
            if all(value is None for value in values):
                continue
            axis.plot(
                epochs,
                [float(value) if value is not None else float("nan") for value in values],
                marker="o",
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(metric_name)
        axis.set_xticks(epochs)
        axis.grid(True, alpha=0.25)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=min(len(legend_labels), 7),
            fontsize=8,
            framealpha=0.9,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved per-class metric plots to {plot_path}")


def configure_matplotlib_cache(plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = plot_path.parent / ".matplotlib"
    cache_dir = plot_path.parent / ".cache"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def collect_class_labels(history: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for epoch_metrics in history:
        labels.update(epoch_metrics.get("eval_per_class_metrics", {}).keys())
    return sorted(labels)


def get_class_metric(
    epoch_metrics: dict[str, Any],
    label: str,
    metric_name: str,
) -> float | None:
    class_metrics = epoch_metrics.get("eval_per_class_metrics", {}).get(label)
    if class_metrics is None:
        return None
    value = class_metrics.get(metric_name)
    return float(value) if value is not None else None


def pick_confusion_matrix(
    final_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, dict[str, int]] | None:
    if "test_confusion_matrix" in final_metrics:
        return final_metrics["test_confusion_matrix"]
    if "eval_confusion_matrix" in final_metrics:
        return final_metrics["eval_confusion_matrix"]
    for item in reversed(history):
        if "eval_confusion_matrix" in item:
            return item["eval_confusion_matrix"]
    return None


def plot_confusion_matrix(
    axis: Any,
    confusion_matrix: dict[str, dict[str, int]] | None,
    title: str,
) -> None:
    axis.set_title(title)
    if not confusion_matrix:
        axis.text(0.5, 0.5, "no confusion matrix", ha="center", va="center")
        return

    labels = list(confusion_matrix)
    matrix = [
        [confusion_matrix[true_label][predicted_label] for predicted_label in labels]
        for true_label in labels
    ]
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")

    max_value = max(max(row) for row in matrix) if matrix else 0
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=color)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


if __name__ == "__main__":
    main()
