"""Base layer class."""

import numpy as np


class Layer:
    """Base class for all neural network layers."""

    def forward(self, input):
        """
        Forward pass.

        Args:
            input: Input data

        Returns:
            Output of the layer
        """
        return input

    def backward(self, grad_output):
        """
        Backward pass.

        Args:
            grad_output: Gradient from the next layer

        Returns:
            Gradient with respect to input
        """
        return grad_output
