import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# SESSION 4 CODE (Base - unchanged)
# =============================================================================


def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    """Load MNIST dataset from local CSV files."""
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    y_train_full = train_data.iloc[:, 0].to_numpy(np.int64)
    X_train_full = train_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    y_test = test_data.iloc[:, 0].to_numpy(np.int64)
    X_test = test_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    n_val = int(len(X_train_full) * val_split)
    np.random.seed(42)
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)

    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False

    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, grad_output):
        return grad_output

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        relu_grad = self.input > 0
        return grad_output * relu_grad


class Dense(Layer):
    """Fully connected layer."""

    def __init__(self, input_units, output_units):
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.biases = np.zeros(output_units)
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output):
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)

        # Store gradients on the layer; the optimizer updates parameters.
        self.grad_weights = grad_weights
        self.grad_biases = grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, labels):
    """Softmax cross-entropy loss and gradient."""
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
    grad = (softmax_probs - one_hot_labels) / batch_size

    return loss, grad


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def forward(network, X):
    """Forward pass through all layers."""
    layer_inputs = []
    activations = []
    input = X
    for layer in network:
        layer_inputs.append(input)
        input = layer.forward(input)
        activations.append(input)
    return layer_inputs, activations

def predict(network, X):
    """Get class predictions."""
    logits = forward(network, X)[1][-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)

# =============================================================================
# SESSION 5 CODE (Extension - Optimizers)
# =============================================================================


class Optimizer:
    def __init__(self, params, learning_rate=0.1):
        self.params = list(params)
        self.learning_rate = learning_rate

    def zero_grad(self):
        for layer in self.params:
            layer.grad_weights = np.zeros_like(layer.weights)
            layer.grad_biases = np.zeros_like(layer.biases)

    def step(self):
        raise NotImplementedError()


class GD(Optimizer):
    """
    Gradient Descent optimizer.
    Full-batch Gradient Descent: one parameter update per epoch on the full dataset.
    """
    batch_size = None
    def __init__(self, params, learning_rate=0.1):
        super().__init__(params=params, learning_rate=learning_rate)

    def step(self):
        for layer in self.params:
            if layer.grad_weights is not None:
                layer.weights = layer.weights - self.learning_rate * layer.grad_weights
                layer.biases = layer.biases - self.learning_rate * layer.grad_biases


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    Mini-batch Stochastic Gradient Descent.
    """
    def __init__(self, params, learning_rate=0.01, batch_size=64):
        super().__init__(params=params, learning_rate=learning_rate)
        self.batch_size = batch_size

    def step(self):
        for layer in self.params:
            if layer.grad_weights is not None:
                layer.weights = layer.weights - self.learning_rate * layer.grad_weights
                layer.biases = layer.biases - self.learning_rate * layer.grad_biases


class Momentum(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9, batch_size=64):
        super().__init__(params=params, learning_rate=learning_rate)
        self.momentum = momentum
        self.batch_size = batch_size
        self.v_weights = {}
        self.v_biases = {}

    def step(self):
        for layer in self.params:
            if layer.grad_weights is None:
                continue

            key = id(layer)
            if key not in self.v_weights:
                self.v_weights[key] = np.zeros_like(layer.grad_weights)
                self.v_biases[key] = np.zeros_like(layer.grad_biases)

            self.v_weights[key] = (
                self.momentum * self.v_weights[key]
                + (1 - self.momentum) * layer.grad_weights
            )
            self.v_biases[key] = (
                self.momentum * self.v_biases[key]
                + (1 - self.momentum) * layer.grad_biases
            )

            layer.weights = layer.weights - self.learning_rate * self.v_weights[key]
            layer.biases = layer.biases - self.learning_rate * self.v_biases[key]


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    Like SGD, but with momentum + adaptive learning rates.
    """
    def __init__(
        self,
        params,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        batch_size=64,
    ):
        super().__init__(params=params, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.t = 0

        # Moment estimates keyed by *layer identity*.
        # Using id(weights) breaks because we replace the arrays every update.
        self.m_weights, self.v_weights = {}, {}
        self.m_biases, self.v_biases = {}, {}

    def step(self):
        self.t += 1

        for layer in self.params:
            if layer.grad_weights is None:
                continue

            grad_weights = layer.grad_weights
            grad_biases = layer.grad_biases

            key = id(layer)
            if key not in self.m_weights:
                self.m_weights[key] = np.zeros_like(grad_weights)
                self.v_weights[key] = np.zeros_like(grad_weights)
                self.m_biases[key] = np.zeros_like(grad_biases)
                self.v_biases[key] = np.zeros_like(grad_biases)

            self.m_weights[key] = self.beta1 * self.m_weights[key] + (1 - self.beta1) * grad_weights
            self.m_biases[key] = self.beta1 * self.m_biases[key] + (1 - self.beta1) * grad_biases

            self.v_weights[key] = self.beta2 * self.v_weights[key] + (1 - self.beta2) * (grad_weights**2)
            self.v_biases[key] = self.beta2 * self.v_biases[key] + (1 - self.beta2) * (grad_biases**2)

            m_hat_w = self.m_weights[key] / (1 - self.beta1**self.t)
            v_hat_w = self.v_weights[key] / (1 - self.beta2**self.t)
            m_hat_b = self.m_biases[key] / (1 - self.beta1**self.t)
            v_hat_b = self.v_biases[key] / (1 - self.beta2**self.t)

            layer.weights = layer.weights - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.biases = layer.biases - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class Model:
    def __init__(self, network):
        self.network = network

    def parameters(self):
        return [layer for layer in self.network if isinstance(layer, Dense)]

    def __call__(self, X):
        activations = []
        input = X
        for layer in self.network:
            input = layer.forward(input)
            activations.append(input)
        return activations[-1]

    def backward(self, grad_logits):
        grad = grad_logits
        for i in range(len(self.network) - 1, -1, -1):
            grad = self.network[i].backward(grad)
        return grad


class _Loss:
    def __init__(self, value, grad_logits, model):
        self._value = float(value)
        self._grad_logits = grad_logits
        self._model = model

    def item(self):
        return self._value

    def backward(self):
        if self._model is None:
            raise ValueError("Loss.backward() was called without a model")
        self._model.backward(self._grad_logits)


class CrossEntropyLoss:
    def __call__(self, logits, y, model=None):
        value, grad_logits = softmax_crossentropy_with_logits(logits, y)
        return _Loss(value, grad_logits, model)


def create_network(
    optimizer_type="sgd", learning_rate=0.01, batch_size=64, momentum=0.9
):
    """
    Create a network with specified optimizer.
    """
    network = [
        Dense(784, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 10),
    ]

    model = Model(network)
    params = model.parameters()

    if optimizer_type == "gd":
        optimizer = GD(params=params, learning_rate=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = SGD(
            params=params, learning_rate=learning_rate, batch_size=batch_size
        )
    elif optimizer_type == "momentum":
        optimizer = Momentum(
            params=params,
            learning_rate=learning_rate,
            momentum=momentum,
            batch_size=batch_size,
        )
    elif optimizer_type == "adam":
        optimizer = Adam(
            params=params, learning_rate=learning_rate, batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return model, optimizer


def iterate_minibatches(X, y, batch_size):
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def train_network(model, optimizer, X_train, y_train, X_val, y_val, num_epochs=10):
    """
    Training loop - clean and optimizer-agnostic.
    Batching is handled here; the optimizer applies updates via optimizer.step().
    """
    history = {"loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    print(f"Training with {type(optimizer).__name__}")
    if isinstance(optimizer, GD):
        batch_size = None
        print(f"  Full batch (Gradient Descent)")
    else:
        batch_size = optimizer.batch_size
        print(f"  Batch size: {batch_size}")

    criterion = CrossEntropyLoss()

    for epoch in range(num_epochs):
        if batch_size:
            batch_losses = []
            for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size=batch_size):
                logits = model(X_batch)
                loss = criterion(logits, y_batch, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            loss_value = float(np.mean(batch_losses)) if batch_losses else float("nan")
        else:
            logits = model(X_train)
            loss = criterion(logits, y_train, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

        history["loss"].append(loss_value)

        val_logits = model(X_val)
        val_loss_value = criterion(val_logits, y_val, model).item()
        history["val_loss"].append(val_loss_value)

        train_preds = predict(model.network, X_train)
        train_acc = np.mean(train_preds == y_train)
        history["train_accuracy"].append(train_acc)

        # Validation
        val_preds = predict(model.network, X_val)
        val_acc = np.mean(val_preds == y_val)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {loss_value:.4f} | Val Loss: {val_loss_value:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

    return history


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load MNIST from local CSV files
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
        "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
    )

    print("\n" + "=" * 60)
    print("TRAINING WITH DIFFERENT OPTIMIZERS")
    print("=" * 60)

    num_epochs = 100

    # Train with GD
    print("\n1. Gradient Descent (full batch)")
    model_gd, opt_gd = create_network(optimizer_type='gd', learning_rate=0.05)
    history_gd = train_network(model_gd, opt_gd, X_train, y_train,
                                X_val, y_val, num_epochs=num_epochs)
    acc_gd = np.mean(predict(model_gd.network, X_test) == y_test)
    print(f"Test Accuracy: {acc_gd*100:.2f}%")

    # Train with SGD
    print("\n2. Stochastic Gradient Descent (batch_size=64)")
    model_sgd, opt_sgd = create_network(optimizer_type='sgd', learning_rate=0.01, batch_size=64)
    history_sgd = train_network(model_sgd, opt_sgd, X_train, y_train,
                                 X_val, y_val, num_epochs=num_epochs)
    acc_sgd = np.mean(predict(model_sgd.network, X_test) == y_test)
    print(f"Test Accuracy: {acc_sgd*100:.2f}%")

    # Train with Adam
    print("\n3. Adam (batch_size=64)")
    model_adam, opt_adam = create_network(optimizer_type='adam', learning_rate=0.001, batch_size=64)
    history_adam = train_network(model_adam, opt_adam, X_train, y_train,
                                  X_val, y_val, num_epochs=num_epochs)
    acc_adam = np.mean(predict(model_adam.network, X_test) == y_test)
    print(f"Test Accuracy: {acc_adam*100:.2f}%")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"GD:   {acc_gd*100:.2f}%")
    print(f"SGD:  {acc_sgd*100:.2f}%")
    print(f"Adam: {acc_adam*100:.2f}%")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_gd['loss'], label='GD')
    plt.plot(history_sgd['loss'], label='SGD')
    plt.plot(history_adam['loss'], label='Adam')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([a*100 for a in history_gd['val_accuracy']], label='GD')
    plt.plot([a*100 for a in history_sgd['val_accuracy']], label='SGD')
    plt.plot([a*100 for a in history_adam['val_accuracy']], label='Adam')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
