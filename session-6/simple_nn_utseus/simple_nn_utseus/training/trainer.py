"""Training utilities for neural networks."""

import numpy as np
from ..losses.crossentropy import softmax_crossentropy_with_logits, softmax


def forward(network, X):
    """
    Perform forward pass through the network.

    Args:
        network: List of layer objects
        X: Input data

    Returns:
        List of activations from each layer
    """
    activations = []
    input_data = X

    for layer in network:
        input_data = layer.forward(input_data)
        activations.append(input_data)

    return activations


def predict(network, X):
    """
    Get class predictions from the network.

    Args:
        network: List of layer objects
        X: Input data

    Returns:
        Predicted class labels
    """
    logits = forward(network, X)[-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)


def train(network, X, y):
    """
    Train the network on a batch of data.

    Args:
        network: List of layer objects
        X: Input data
        y: True labels

    Returns:
        Loss value
    """
    # Forward pass
    activations = forward(network, X)
    logits = activations[-1]

    # Compute loss and gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    # Backward pass
    grad_output = grad_logits
    for layer in reversed(network):
        grad_output = layer.backward(grad_output)

    return loss


def train_network(network, X_train, y_train, X_val, y_val, num_epochs=200):
    """
    Full training loop for a neural network.

    Args:
        network: List of layer objects
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_epochs: Number of training epochs
    """
    print(f"Training on {len(X_train)} examples using Gradient Descent")

    for epoch in range(num_epochs):
        # Train on entire dataset (full batch GD)
        loss = train(network, X_train, y_train)

        # Compute training accuracy
        train_predictions = predict(network, X_train)
        train_accuracy = np.mean(train_predictions == y_train)

        # Compute validation accuracy
        val_predictions = predict(network, X_val)
        val_accuracy = np.mean(val_predictions == y_val)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, "
            f"Val Acc: {val_accuracy:.4f}"
        )

