"""General utility functions."""

import numpy as np


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def print_network_architecture(network):
    """
    Print network architecture summary.

    Args:
        network: List of layer objects
    """
    print("Network Architecture:")
    print("-" * 50)
    for i, layer in enumerate(network):
        layer_type = layer.__class__.__name__
        if hasattr(layer, "weights"):
            shape = f"{layer.weights.shape[0]} → {layer.weights.shape[1]}"
            print(f"Layer {i}: {layer_type:15} ({shape})")
        else:
            print(f"Layer {i}: {layer_type:15}")
    print("-" * 50)
