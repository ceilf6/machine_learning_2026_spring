"""Fully connected (Dense) layer implementation."""
import numpy as np
from .base import Layer


class Dense(Layer):
    """Fully connected layer with learnable weights and biases."""
    
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        Initialize Dense layer.
        
        Args:
            input_units: Number of input features
            output_units: Number of output features
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # He initialization for better training with ReLU
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
        self.biases = np.zeros(output_units)
    
    def forward(self, input):
        """Compute forward pass: output = input @ weights + biases."""
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, grad_output):
        """
        Compute backward pass and update parameters.
        
        Args:
            grad_output: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
        """
        # Compute gradients
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Update parameters
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        
        return grad_input