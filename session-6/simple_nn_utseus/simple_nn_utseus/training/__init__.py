"""
Training utilities and loops.

This subpackage provides functions for training neural networks.
"""

from .trainer import forward, predict, train, train_network

__all__ = [
    "forward",
    "predict",
    "train",
    "train_network",
]

# _metrics is internal and not exported
