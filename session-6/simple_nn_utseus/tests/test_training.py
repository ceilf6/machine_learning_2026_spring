"""Tests for training utilities."""

import numpy as np
from simple_nn_utseus.layers import Dense, ReLU
from simple_nn_utseus.training import forward, predict, train


class TestTraining:
    """Test training utilities."""

    def test_forward_pass(self):
        """Test forward pass through network."""
        network = [Dense(10, 5), ReLU(), Dense(5, 3)]
        x = np.random.randn(32, 10)
        activations = forward(network, x)

        # Should have one activation per layer
        assert len(activations) == 3
        # Final output should have correct shape
        assert activations[-1].shape == (32, 3)

    def test_predict(self):
        """Test prediction function."""
        network = [Dense(10, 5), ReLU(), Dense(5, 3)]
        x = np.random.randn(32, 10)
        predictions = predict(network, x)

        # Should return class indices
        assert predictions.shape == (32,)
        assert predictions.dtype in [np.int32, np.int64]
        # All predictions should be valid class indices (0, 1, or 2)
        assert all(0 <= p < 3 for p in predictions)

    def test_train_reduces_loss(self):
        """Test that training reduces loss over time."""
        # Simple problem: learn identity-like mapping
        np.random.seed(42)
        network = [Dense(10, 10, learning_rate=0.1)]

        # Generate simple data
        X = np.eye(10)  # Identity matrix
        y = np.arange(10)  # Labels 0-9

        # Train for several iterations
        losses = []
        for _ in range(100):
            loss = train(network, X, y)
            losses.append(loss)

        # Loss should generally decrease (check first vs last)
        assert losses[-1] < losses[0], "Loss should decrease during training"
