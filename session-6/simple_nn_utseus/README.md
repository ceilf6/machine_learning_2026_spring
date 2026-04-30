# Simple Neural Network UTSEUS (simple_nn_utseus)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()



A simple, educational neural network library implemented from scratch using NumPy.

## Features

- 🧠 **Custom Neural Network Layers**: Dense, ReLU, Sigmoid
- 📊 **MNIST Data Loading**: Easy CSV-based data loading
- 🎯 **Training Utilities**: Forward/backward propagation, gradient descent
- 📦 **Modular Design**: Clean separation of concerns with subpackages
- 📚 **Educational Focus**: Clear, readable code for learning purposes

## Installation

### From Source (for development)

```bash
git clone https://github.com/yourusername/simple_nn_utseus.git
cd simple_nn_utseus
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from simple_nn_utseus import (
    load_mnist_from_csv,
    Dense,
    ReLU,
    train_network,
    print_network_architecture
)

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
    "mnist_train.csv",
    "mnist_test.csv",
    val_split=0.1
)

# Build network
network = [
    Dense(784, 64),   # Input layer
    ReLU(),           # Activation
    Dense(64, 32),    # Hidden layer
    ReLU(),           # Activation
    Dense(32, 10)     # Output layer
]

# Print architecture
print_network_architecture(network)

# Train
train_network(network, X_train, y_train, X_val, y_val, num_epochs=50)
```

## Package Structure

```
simple_nn_utseus/
├── data/          # Data loading utilities
├── layers/        # Neural network layers
├── losses/        # Loss functions
├── training/      # Training loops and utilities
└── utils/         # Helper functions
```

## API Overview

### Data Loading

- `load_mnist_from_csv()`: Load MNIST dataset from CSV files

### Layers

- `Layer`: Base layer class
- `Dense`: Fully connected layer
- `ReLU`: ReLU activation
- `Sigmoid`: Sigmoid activation

### Losses

- `softmax_crossentropy_with_logits()`: Softmax + cross-entropy loss
- `softmax()`: Softmax function

### Training

- `forward()`: Forward pass through network
- `predict()`: Get predictions
- `train()`: Single training step
- `train_network()`: Full training loop

### Utilities

- `set_random_seed()`: Set random seed for reproducibility
- `print_network_architecture()`: Display network structure

## Advanced Usage

### Custom Training Loop

```python
from simple_nn_utseus import Dense, ReLU, forward, train, predict

network = [Dense(784, 64), ReLU(), Dense(64, 10)]

for epoch in range(100):
    loss = train(network, X_train, y_train)
    predictions = predict(network, X_val)
    accuracy = (predictions == y_val).mean()
    print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={accuracy:.4f}")
```

### Accessing Subpackages

```python
# Import from specific subpackages
from simple_nn_utseus.layers import Dense, ReLU
from simple_nn_utseus.losses import softmax_crossentropy_with_logits
from simple_nn_utseus.training import train_network
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black simple_nn_utseus/
```

## License

MIT License - See LICENSE file for details

## Contributing

This is an educational project. Contributions are welcome for:
- Bug fixes
- Documentation improvements
- Additional layer types
- More loss functions

## Acknowledgments

Built for educational purposes to demonstrate Python packaging concepts.