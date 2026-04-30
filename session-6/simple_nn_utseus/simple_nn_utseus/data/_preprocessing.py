"""Internal preprocessing utilities (not exposed to users)."""

import numpy as np


def _normalize_images(images):
    """Internal function to normalize image data."""
    return images / 255.0


def _create_validation_split(X, y, val_split=0.1, seed=42):
    """Internal function to create validation split."""
    n_val = int(len(X) * val_split)
    np.random.seed(seed)
    val_indices = np.random.choice(len(X), n_val, replace=False)
    train_mask = np.ones(len(X), dtype=bool)
    train_mask[val_indices] = False
    return X[train_mask], y[train_mask], X[val_indices], y[val_indices]
