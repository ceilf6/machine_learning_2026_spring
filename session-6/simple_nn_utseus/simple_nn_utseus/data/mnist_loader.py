"""MNIST data loading utilities."""

import pandas as pd
import numpy as np


def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    """
    Load MNIST dataset from CSV files and split into train/val/test sets.

    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to test CSV file
        val_split: Fraction of training data to use for validation

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    y_train_full = train_data.iloc[:, 0].to_numpy(np.int64)
    X_train_full = train_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    y_test = test_data.iloc[:, 0].to_numpy(np.int64)
    X_test = test_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    # Split validation set
    n_val = int(len(X_train_full) * val_split)
    np.random.seed(42)
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)

    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False

    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test
