"""Cross-entropy loss implementations."""

import numpy as np


def softmax_crossentropy_with_logits(logits, labels):
    """
    Compute softmax cross-entropy loss and gradient.

    Args:
        logits: Raw network outputs (before softmax), shape (batch_size, num_classes)
        labels: True class labels, shape (batch_size,)

    Returns:
        Tuple of (loss, gradient):
            - loss: Scalar loss value
            - gradient: Gradient of loss with respect to logits
    """
    batch_size = logits.shape[0]

    # Create one-hot encoded labels
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    # Compute softmax with numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy loss
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size

    # Gradient of cross-entropy with respect to logits
    grad = (softmax_probs - one_hot_labels) / batch_size

    return loss, grad


def softmax(logits):
    """
    Compute softmax probabilities.

    Args:
        logits: Raw scores, shape (batch_size, num_classes)

    Returns:
        Probability distribution over classes
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
