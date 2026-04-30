"""
Simple Neural Network Package (simple_nn_utseus)

A simple, educational neural network library implemented from scratch in NumPy.

Main Components:
    - data: Data loading utilities
    - layers: Neural network layer implementations
    - losses: Loss functions
    - training: Training utilities
    - utils: General helper functions

Example:
    >>> from simple_nn_utseus import Dense, ReLU, load_mnist_from_csv, train_network
    >>> X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
    ...     "mnist_train.csv", "mnist_test.csv"
    ... )
    >>> network = [Dense(784, 64), ReLU(), Dense(64, 10)]
    >>> train_network(network, X_train, y_train, X_val, y_val, num_epochs=10)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import from subpackages - expose the most commonly used items
from .data import load_mnist_from_csv
from .layers import Layer, Dense, ReLU, Sigmoid
from .losses import softmax_crossentropy_with_logits, softmax
from .training import forward, predict, train, train_network
from .utils import set_random_seed, print_network_architecture

# Define public API
__all__ = [
    # Data utilities
    "load_mnist_from_csv",
    # Layers
    "Layer",
    "Dense",
    "ReLU",
    "Sigmoid",
    # Losses
    "softmax_crossentropy_with_logits",
    "softmax",
    # Training
    "forward",
    "predict",
    "train",
    "train_network",
    # Utils
    "set_random_seed",
    "print_network_architecture",
]
