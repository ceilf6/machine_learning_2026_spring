"""Internal metrics computation (not exposed to users)."""

import numpy as np


def _compute_accuracy(predictions, labels):
    """Internal: Compute classification accuracy."""
    return np.mean(predictions == labels)


def _compute_confusion_matrix(predictions, labels, num_classes):
    """Internal: Compute confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        matrix[true, pred] += 1
    return matrix
