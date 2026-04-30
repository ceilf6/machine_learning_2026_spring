"""
Example: Training a neural network on MNIST dataset.

This script demonstrates how to use the simple_nn_utseus package
to build and train a neural network on the MNIST dataset.
"""

from simple_nn_utseus import (
    load_mnist_from_csv,
    Dense,
    ReLU,
    train_network,
    print_network_architecture,
    set_random_seed,
)


def main():
    """Main training function."""
    # Set random seed for reproducibility
    set_random_seed(42)

    # Load MNIST data
    print("Loading MNIST data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
        "mnist_train.csv", "mnist_test.csv", val_split=0.1
    )

    # Build network architecture
    network = [
        Dense(784, 128, learning_rate=0.1),  # Input: 784 (28x28 images)
        ReLU(),
        Dense(128, 64, learning_rate=0.1),
        ReLU(),
        Dense(64, 10, learning_rate=0.1),  # Output: 10 classes (0-9)
    ]

    # Print network structure
    print_network_architecture(network)

    # Train the network
    print("\nStarting training...")
    train_network(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=50,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
