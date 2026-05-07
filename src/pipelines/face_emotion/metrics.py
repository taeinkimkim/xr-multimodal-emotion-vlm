"""Evaluation metrics for face emotion classification."""

from __future__ import annotations

import torch


def new_evaluation_stats(num_labels: int, ece_bins: int) -> dict[str, torch.Tensor]:
    return {
        "support": torch.zeros(num_labels, dtype=torch.long),
        "correct": torch.zeros(num_labels, dtype=torch.long),
        "loss_sum": torch.zeros(num_labels, dtype=torch.float64),
        "true_probability_sum": torch.zeros(num_labels, dtype=torch.float64),
        "predicted_probability_sum": torch.zeros(num_labels, dtype=torch.float64),
        "softmax_sum": torch.zeros(num_labels, num_labels, dtype=torch.float64),
        "confusion_matrix": torch.zeros(num_labels, num_labels, dtype=torch.long),
        "confidence_sum": torch.zeros(1, dtype=torch.float64),
        "correct_confidence_sum": torch.zeros(1, dtype=torch.float64),
        "incorrect_confidence_sum": torch.zeros(1, dtype=torch.float64),
        "correct_count": torch.zeros(1, dtype=torch.long),
        "incorrect_count": torch.zeros(1, dtype=torch.long),
        "ece_confidence_sum": torch.zeros(ece_bins, dtype=torch.float64),
        "ece_correct_sum": torch.zeros(ece_bins, dtype=torch.float64),
        "ece_count": torch.zeros(ece_bins, dtype=torch.long),
    }


def update_class_stats(
    stats: dict[str, torch.Tensor],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    sample_losses: torch.Tensor,
) -> None:
    labels_cpu = labels.detach().cpu()
    predictions_cpu = predictions.detach().cpu()
    probabilities_cpu = probabilities.detach().cpu().to(torch.float64)
    sample_losses_cpu = sample_losses.detach().cpu().to(torch.float64)

    for index, label in enumerate(labels_cpu):
        label_id = int(label.item())
        prediction_id = int(predictions_cpu[index].item())
        stats["support"][label_id] += 1
        stats["correct"][label_id] += int(prediction_id == label_id)
        stats["loss_sum"][label_id] += sample_losses_cpu[index]
        stats["true_probability_sum"][label_id] += probabilities_cpu[index, label_id]
        stats["predicted_probability_sum"][label_id] += probabilities_cpu[index, prediction_id]
        stats["softmax_sum"][label_id] += probabilities_cpu[index]
        stats["confusion_matrix"][label_id, prediction_id] += 1

        confidence = probabilities_cpu[index, prediction_id]
        is_correct = prediction_id == label_id
        stats["confidence_sum"][0] += confidence
        if is_correct:
            stats["correct_confidence_sum"][0] += confidence
            stats["correct_count"][0] += 1
        else:
            stats["incorrect_confidence_sum"][0] += confidence
            stats["incorrect_count"][0] += 1

        ece_bins = stats["ece_count"].numel()
        bin_index = min(int(confidence.item() * ece_bins), ece_bins - 1)
        stats["ece_confidence_sum"][bin_index] += confidence
        stats["ece_correct_sum"][bin_index] += float(is_correct)
        stats["ece_count"][bin_index] += 1


def finalize_evaluation_metrics(
    stats: dict[str, torch.Tensor],
    id2label: dict[int, str],
    top_confused_pairs: int,
) -> dict[str, object]:
    per_class_metrics: dict[str, dict[str, object]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    weighted_f1_sum = 0.0
    total_support = int(stats["support"].sum().item())

    for label_id in sorted(id2label):
        label = id2label[label_id]
        support = int(stats["support"][label_id].item())
        if support == 0:
            continue

        true_positive = int(stats["confusion_matrix"][label_id, label_id].item())
        predicted_positive = int(stats["confusion_matrix"][:, label_id].sum().item())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        f1_scores.append(f1)
        weighted_f1_sum += f1 * support

        per_class_metrics[label] = {
            "support": support,
            "loss": float(stats["loss_sum"][label_id].item() / support),
            "accuracy": float(stats["correct"][label_id].item() / support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_true_class_softmax": float(stats["true_probability_sum"][label_id].item() / support),
            "avg_predicted_softmax": float(stats["predicted_probability_sum"][label_id].item() / support),
            "mean_softmax": {
                id2label[index]: float(value.item() / support)
                for index, value in enumerate(stats["softmax_sum"][label_id])
            },
        }

    confusion_matrix = format_confusion_matrix(stats["confusion_matrix"], id2label)
    return {
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "weighted_f1": weighted_f1_sum / total_support if total_support else 0.0,
        "mean_confidence": safe_div(float(stats["confidence_sum"][0].item()), total_support),
        "correct_confidence": safe_div(
            float(stats["correct_confidence_sum"][0].item()),
            int(stats["correct_count"][0].item()),
        ),
        "incorrect_confidence": safe_div(
            float(stats["incorrect_confidence_sum"][0].item()),
            int(stats["incorrect_count"][0].item()),
        ),
        "ece": compute_ece(stats),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "top_confused_pairs": top_confused_pairs_from_matrix(
            stats["confusion_matrix"],
            id2label,
            top_k=top_confused_pairs,
        ),
    }


def prefix_metrics(metrics: dict[str, object], prefix: str) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def print_evaluation_metrics(metrics: dict[str, object], split_name: str) -> None:
    print(f"[{split_name} metrics]")
    print(
        f"  loss={metrics['loss']:.4f}\n"
        f"  accuracy={metrics['accuracy']:.2%}\n"
        f"  balanced_accuracy={metrics['balanced_accuracy']:.2%}\n"
        f"  macro_f1={metrics['macro_f1']:.4f}\n"
        f"  weighted_f1={metrics['weighted_f1']:.4f}\n"
        f"  mean_confidence={metrics['mean_confidence']:.4f}\n"
        f"  correct_confidence={metrics['correct_confidence']:.4f}\n"
        f"  incorrect_confidence={metrics['incorrect_confidence']:.4f}\n"
        f"  ece={metrics['ece']:.4f}"
    )
    print(f"\n[{split_name} per-class metrics]")
    print("label      support  precision  recall  f1      accuracy  loss")
    print("---------  -------  ---------  ------  ------  --------  ------")
    for label, class_metrics in metrics["per_class_metrics"].items():
        print(
            f"{label:<9}  "
            f"{class_metrics['support']:>7}  "
            f"{class_metrics['precision']:>9.2%}  "
            f"{class_metrics['recall']:>6.2%}  "
            f"{class_metrics['f1']:>6.4f}  "
            f"{class_metrics['accuracy']:>8.2%}  "
            f"{class_metrics['loss']:>6.4f}"
        )
    print_confusion_matrix(metrics["confusion_matrix"], split_name=split_name)
    print(f"\n[{split_name} top confused pairs]")
    for pair in metrics["top_confused_pairs"]:
        print(
            "   "
            f"{pair['true_label']} -> {pair['predicted_label']}: "
            f"{pair['count']} ({pair['rate_within_true_label']:.2%})"
        )


def print_confusion_matrix(
    confusion_matrix: dict[str, dict[str, int]],
    split_name: str,
) -> None:
    labels = list(confusion_matrix)
    column_width = max(7, max(len(label) for label in labels))
    print(f"\n[{split_name} confusion matrix]")
    print("true\\pred".ljust(column_width), end="")
    for label in labels:
        print(f"  {label:>{column_width}}", end="")
    print()
    for true_label in labels:
        print(f"{true_label:<{column_width}}", end="")
        for predicted_label in labels:
            print(f"  {confusion_matrix[true_label][predicted_label]:>{column_width}}", end="")
        print()


def safe_div(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_ece(stats: dict[str, torch.Tensor]) -> float:
    total = int(stats["ece_count"].sum().item())
    if total == 0:
        return 0.0

    ece = 0.0
    for index, count_tensor in enumerate(stats["ece_count"]):
        count = int(count_tensor.item())
        if count == 0:
            continue
        average_confidence = float(stats["ece_confidence_sum"][index].item() / count)
        average_accuracy = float(stats["ece_correct_sum"][index].item() / count)
        ece += (count / total) * abs(average_accuracy - average_confidence)
    return ece


def format_confusion_matrix(
    confusion_matrix: torch.Tensor,
    id2label: dict[int, str],
) -> dict[str, dict[str, int]]:
    return {
        id2label[row_index]: {
            id2label[column_index]: int(value.item())
            for column_index, value in enumerate(row)
        }
        for row_index, row in enumerate(confusion_matrix)
    }


def top_confused_pairs_from_matrix(
    confusion_matrix: torch.Tensor,
    id2label: dict[int, str],
    top_k: int,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for true_index, row in enumerate(confusion_matrix):
        true_count = int(row.sum().item())
        for predicted_index, value in enumerate(row):
            if true_index == predicted_index:
                continue
            count = int(value.item())
            if count == 0:
                continue
            pairs.append(
                {
                    "true_label": id2label[true_index],
                    "predicted_label": id2label[predicted_index],
                    "count": count,
                    "rate_within_true_label": count / true_count if true_count else 0.0,
                }
            )
    return sorted(pairs, key=lambda pair: pair["count"], reverse=True)[:top_k]
