"""Small, dependency-free multilabel AP and threshold metrics."""

from __future__ import annotations

import numpy as np

from src.presence.labels import CLASS_NAMES


def average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Area under the stepwise precision-recall curve, grouping score ties."""
    positive = int(y_true.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    labels = y_true[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    true_positive = np.cumsum(labels)[ends]
    precision = true_positive / (ends + 1)
    gained_positive = np.diff(np.r_[0, true_positive])
    return float(np.sum(precision * gained_positive) / positive)


def foreground_macro_ap(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    scores = [average_precision(y_true[:, i], probabilities[:, i]) for i in range(1, 8)]
    if not np.isfinite(scores).all():
        raise ValueError("Every foreground class must have positive validation samples")
    return float(np.mean(scores))


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Choose a threshold using validation labels only; ties prefer higher threshold."""
    positive = int(y_true.sum())
    if positive == 0:
        return 1.0
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    labels = y_true[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    true_positive = np.cumsum(labels)[ends]
    f1 = 2 * true_positive / (ends + 1 + positive)
    return float(scores[ends[int(np.argmax(f1))]])


def evaluate(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict[str, object]:
    """Report each class and the seven-foreground macro averages."""
    if y_true.shape != probabilities.shape or y_true.shape[1] != 8:
        raise ValueError("Expected matching [N, 8] labels and probabilities")
    per_class: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(CLASS_NAMES):
        truth = y_true[:, i].astype(bool)
        predicted = probabilities[:, i] >= thresholds[i]
        tp = int(np.count_nonzero(truth & predicted))
        fp = int(np.count_nonzero(~truth & predicted))
        fn = int(np.count_nonzero(truth & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[name] = {
            "ap": average_precision(y_true[:, i], probabilities[:, i]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": float(thresholds[i]),
            "support": int(truth.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    foreground = list(CLASS_NAMES[1:])
    return {
        "count": len(y_true),
        "foreground_macro_ap": float(np.mean([per_class[name]["ap"] for name in foreground])),
        "foreground_macro_f1": float(np.mean([per_class[name]["f1"] for name in foreground])),
        "foreground_macro_recall": float(np.mean([per_class[name]["recall"] for name in foreground])),
        "per_class": per_class,
    }
