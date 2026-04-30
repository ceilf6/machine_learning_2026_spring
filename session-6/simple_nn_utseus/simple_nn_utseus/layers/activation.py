"""Activation function layers."""

import numpy as np
from .base import Layer


class ReLU(Layer):
    """Rectified Linear Unit activation function."""

    def forward(self, input):
        """Apply ReLU: max(0, x)."""
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        """
        Compute gradient: 1 if x > 0, else 0.

        Args:
            grad_output: Gradient from the next layer

        Returns:
            Gradient multiplied by ReLU derivative
        """
        relu_grad = self.input > 0
        return grad_output * relu_grad


class Sigmoid(Layer):
    """Sigmoid activation function (example of additional functionality)."""

    def forward(self, input):
        """Apply sigmoid: 1 / (1 + exp(-x))."""
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, grad_output):
        """Compute gradient: sigmoid(x) * (1 - sigmoid(x))."""
        return grad_output * self.output * (1 - self.output)
