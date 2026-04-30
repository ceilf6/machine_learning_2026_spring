"""
Data loading and preprocessing utilities.

This subpackage provides functions for loading datasets like MNIST.
"""

# Export public API
from .mnist_loader import load_mnist_from_csv

# Define what gets imported with "from simple_nn_utseus.data import *"
__all__ = [
    "load_mnist_from_csv",
]

# Note: _preprocessing module is NOT exported (it's internal)
