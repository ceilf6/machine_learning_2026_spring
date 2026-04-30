"""Tests for data loading."""

import numpy as np
import pandas as pd
import tempfile
import os
from simple_nn_utseus.data import load_mnist_from_csv


class TestDataLoading:
    """Test data loading functions."""

    def test_load_mnist_from_csv(self):
        """Test MNIST CSV loading with synthetic data."""
        # Create temporary CSV files with synthetic data
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            test_path = os.path.join(tmpdir, "test.csv")

            # Create synthetic training data (100 samples, 784 features)
            train_data = np.random.randint(0, 255, (100, 785))
            train_data[:, 0] = np.random.randint(0, 10, 100)  # Labels
            pd.DataFrame(train_data).to_csv(train_path, index=False, header=False)

            # Create synthetic test data (20 samples)
            test_data = np.random.randint(0, 255, (20, 785))
            test_data[:, 0] = np.random.randint(0, 10, 20)  # Labels
            pd.DataFrame(test_data).to_csv(test_path, index=False, header=False)

            # Load data
            X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
                train_path, test_path, val_split=0.20000
            )
            
            assert X_train.shape[1] == 784  # 784 features

            # Check data range (should be normalized to [0, 1])
            assert X_train.min() >= 0
            assert X_train.max() <= 1

            # Check label range
            assert y_train.min() >= 0
            assert y_train.max() <= 9
