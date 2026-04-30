"""Tests for neural network layers."""

import numpy as np
import pytest
from simple_nn_utseus.layers import Dense, ReLU, Layer


class TestLayer:
    """Test base Layer class."""

    def test_forward(self):
        """Test that base layer returns input unchanged."""
        layer = Layer()
        x = np.array([[1, 2, 3]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, x)

    def test_backward(self):
        """Test that base layer returns gradient unchanged."""
        layer = Layer()
        grad = np.array([[1, 2, 3]])
        output = layer.backward(grad)
        np.testing.assert_array_equal(output, grad)


class TestDense:
    """Test Dense layer."""

    def test_initialization(self):
        """Test Dense layer initialization."""
        layer = Dense(10, 5, learning_rate=0.01)
        assert layer.weights.shape == (10, 5)
        assert layer.biases.shape == (5,)
        assert layer.learning_rate == 0.01

    def test_forward_shape(self):
        """Test forward pass output shape."""
        layer = Dense(10, 5)
        x = np.random.randn(32, 10)  # Batch of 32 samples
        output = layer.forward(x)
        assert output.shape == (32, 5)

    def test_backward_updates_weights(self):
        """Test that backward pass updates weights."""
        layer = Dense(10, 5, learning_rate=0.1)
        x = np.random.randn(32, 10)

        # Forward pass
        output = layer.forward(x)

        # Save original weights
        original_weights = layer.weights.copy()
        original_biases = layer.biases.copy()

        # Backward pass with random gradient
        grad = np.random.randn(32, 5)
        layer.backward(grad)

        # Check that weights changed
        assert not np.allclose(layer.weights, original_weights)
        assert not np.allclose(layer.biases, original_biases)


class TestReLU:
    """Test ReLU activation."""

    def test_forward_positive(self):
        """Test ReLU with positive inputs."""
        layer = ReLU()
        x = np.array([[1, 2, 3]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, x)

    def test_forward_negative(self):
        """Test ReLU with negative inputs."""
        layer = ReLU()
        x = np.array([[-1, -2, -3]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, np.array([[0, 0, 0]]))

    def test_forward_mixed(self):
        """Test ReLU with mixed inputs."""
        layer = ReLU()
        x = np.array([[-1, 2, -3, 4]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, np.array([[0, 2, 0, 4]]))

    def test_backward(self):
        """Test ReLU backward pass."""
        layer = ReLU()
        x = np.array([[-1, 2, -3, 4]])
        layer.forward(x)

        grad = np.array([[1, 1, 1, 1]])
        output_grad = layer.backward(grad)

        # Gradient should be 0 where input was negative, 1 where positive
        expected = np.array([[0, 1, 0, 1]])
        np.testing.assert_array_equal(output_grad, expected)
