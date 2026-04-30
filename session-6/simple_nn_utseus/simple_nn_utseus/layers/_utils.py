"""Internal layer utilities (not exposed to users)."""

import numpy as np


def _he_initialization(input_size, output_size):
    """Internal: He initialization for weights."""
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)


def _xavier_initialization(input_size, output_size):
    """Internal: Xavier initialization for weights."""
    return np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)

