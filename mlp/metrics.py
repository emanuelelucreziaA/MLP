"""Shared metrics for classification tasks."""

from __future__ import annotations

import numpy as np


def prediction_labels(y_pred_proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities/logits to class indices."""
    return np.argmax(y_pred_proba, axis=1)


def compute_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute classification accuracy from true labels and model outputs."""
    y_pred = prediction_labels(y_pred_proba)
    return float(np.mean(y_pred == y_true))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Compute confusion matrix, inferring class count from data when omitted."""
    if num_classes is None:
        max_true = int(np.max(y_true)) if y_true.size > 0 else 0
        max_pred = int(np.max(y_pred)) if y_pred.size > 0 else 0
        num_classes = max(max_true, max_pred) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute accuracy, confusion matrix, per-class precision/recall/F1, and macro averages."""
    y_pred = prediction_labels(y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0.0

    accuracy = float(np.mean(y_pred == y_true))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "macro_precision": float(np.mean(precision)) if num_classes > 0 else 0.0,
        "macro_recall": float(np.mean(recall)) if num_classes > 0 else 0.0,
        "macro_f1": float(np.mean(f1)) if num_classes > 0 else 0.0,
        "y_pred": y_pred,
    }
