import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# SESSION 4 CODE (Base)
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
    """Base class with train/eval mode support for Session 6."""
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, grad_output):
        return grad_output

    def train_mode(self):
        """Set layer to training mode."""
        if hasattr(self, "training"):
            self.training = True

    def eval_mode(self):
        """Set layer to evaluation mode."""
        if hasattr(self, "training"):
            self.training = False


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        relu_grad = self.input > 0
        return grad_output * relu_grad


# =============================================================================
# SESSION 5 CODE (Optimizers)
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
    """Full-batch Gradient Descent."""
    batch_size = None
    def __init__(self, params, learning_rate=0.1):
        super().__init__(params=params, learning_rate=learning_rate)

    def step(self):
        for layer in self.params:
            if layer.grad_weights is not None:
                layer.weights = layer.weights - self.learning_rate * layer.grad_weights
                layer.biases = layer.biases - self.learning_rate * layer.grad_biases


class SGD(Optimizer):
    """Mini-batch Stochastic Gradient Descent."""
    def __init__(self, params, learning_rate=0.01, batch_size=64):
        super().__init__(params=params, learning_rate=learning_rate)
        self.batch_size = batch_size

    def step(self):
        for layer in self.params:
            if layer.grad_weights is not None:
                layer.weights = layer.weights - self.learning_rate * layer.grad_weights
                layer.biases = layer.biases - self.learning_rate * layer.grad_biases


class Momentum(Optimizer):
    """
    Momentum optimizer (Gradient Descent with Momentum).
    Like SGD, but with momentum to accelerate convergence in consistent directions.
    """

    def __init__(self, params, learning_rate=0.01, momentum=0.9, batch_size=64):
        super().__init__(params=params, learning_rate=learning_rate)
        self.momentum = momentum
        self.batch_size = batch_size

        # Velocity buffers for each parameter
        self.v_weights = {}
        self.v_biases = {}

    def step(self):
        for layer in self.params:
            if layer.grad_weights is None:
                continue

            grad_weights = layer.grad_weights
            grad_biases = layer.grad_biases

            key = id(layer)
            if key not in self.v_weights:
                self.v_weights[key] = np.zeros_like(grad_weights)
                self.v_biases[key] = np.zeros_like(grad_biases)

            # Update velocities
            self.v_weights[key] = (
                self.momentum * self.v_weights[key] - self.learning_rate * grad_weights
            )
            self.v_biases[key] = (
                self.momentum * self.v_biases[key] - self.learning_rate * grad_biases
            )

            # Update parameters
            layer.weights = layer.weights + self.v_weights[key]
            layer.biases = layer.biases + self.v_biases[key]


class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates."""
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=64):
        super().__init__(params=params, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.t = 0
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
            self.v_weights[key] = self.beta2 * self.v_weights[key] + (1 - self.beta2) * (grad_weights ** 2)
            self.v_biases[key] = self.beta2 * self.v_biases[key] + (1 - self.beta2) * (grad_biases ** 2)

            m_hat_w = self.m_weights[key] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_weights[key] / (1 - self.beta2 ** self.t)
            m_hat_b = self.m_biases[key] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_biases[key] / (1 - self.beta2 ** self.t)

            layer.weights = layer.weights - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.biases = layer.biases - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


# =============================================================================
# SESSION 6 CODE (Extension - Regularization)
# =============================================================================


class Dropout(Layer):
    """
    Dropout regularization layer.
    Randomly zeros neurons during training with probability p.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, input):
        if self.training:
            # Inverted dropout: scale during training
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape) / (1 - self.p)
            return input * self.mask
        else:
            return input

    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask
        else:
            return grad_output

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False


class Dense(Layer):
    """
    Fully connected layer with L1/L2 regularization support.
    """
    def __init__(self, input_units, output_units, l1=0.0, l2=0.0):
        super().__init__()
        self.l1 = l1  # L1 regularization strength
        self.l2 = l2  # L2 regularization strength
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

        # Add regularization gradients
        if self.l1 > 0:
            grad_weights += self.l1 * np.sign(self.weights)
        if self.l2 > 0:
            grad_weights += self.l2 * 2 * self.weights

        self.grad_weights = grad_weights
        self.grad_biases = grad_biases

        return grad_input

    def l1_penalty(self):
        """Compute L1 penalty for loss."""
        return self.l1 * np.sum(np.abs(self.weights)) if self.l1 > 0 else 0

    def l2_penalty(self):
        """Compute L2 penalty for loss."""
        return self.l2 * np.sum(self.weights ** 2) if self.l2 > 0 else 0


def softmax_crossentropy_with_logits(logits, labels, network=None):
    """Loss function with regularization penalties."""
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
    grad = (softmax_probs - one_hot_labels) / batch_size

    # Add regularization penalties
    if network:
        for layer in network:
            if isinstance(layer, Dense):
                loss += layer.l1_penalty()
                loss += layer.l2_penalty()

    return loss, grad


def set_train_mode(network):
    for layer in network:
        layer.train_mode()


def set_eval_mode(network):
    for layer in network:
        layer.eval_mode()


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def forward(network, X):
    """Forward pass through all layers.

    Important: this function does NOT change train/eval mode.
    The caller is responsible for calling set_train_mode() / set_eval_mode().
    """
    layer_inputs = []
    activations = []
    input = X
    for layer in network:
        layer_inputs.append(input)
        input = layer.forward(input)
        activations.append(input)
    return layer_inputs, activations


def predict(network, X):
    """Get predictions."""
    set_eval_mode(network)
    logits = forward(network, X)[1][-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)


def iterate_minibatches(X, y, batch_size):
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


class Model:
    def __init__(self, network):
        self.network = network

    def parameters(self):
        return [layer for layer in self.network if isinstance(layer, Dense)]

    def train(self):
        set_train_mode(self.network)

    def eval(self):
        set_eval_mode(self.network)

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
        self._model.backward(self._grad_logits)


class CrossEntropyLoss:
    def __call__(self, logits, y, model=None):
        value, grad_logits = softmax_crossentropy_with_logits(logits, y, model.network if model else None)
        return _Loss(value, grad_logits, model)


def train_network(
    model,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs=20,
):
    """Training loop with explicit train/eval mode control (Dropout)."""
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
        model.train()

        if batch_size:
            batch_losses = []
            for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size=batch_size):
                logits = model(X_batch)
                loss = criterion(logits, y_batch, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        else:
            logits = model(X_train)
            loss = criterion(logits, y_train, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        model.eval()

        train_logits = model(X_train)
        train_preds = np.argmax(softmax(train_logits), axis=-1)
        train_acc = np.mean(train_preds == y_train)

        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val, model).item()
        val_preds = np.argmax(softmax(val_logits), axis=-1)
        val_acc = np.mean(val_preds == y_val)

        history["loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train: {train_loss:.4f}/{train_acc*100:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc*100:.1f}%")

    return history


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


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load MNIST from local CSV
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
        "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
    )

    subset_size = min(15000, X_train.shape[0])
    np.random.seed(42)
    indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    print("\n" + "=" * 60)
    print("COMPARING REGULARIZATION TECHNIQUES")
    print("=" * 60)

    num_epochs = 100

    # 1. Baseline
    print("\n1. Baseline (Adam, no regularization)")
    model_baseline, opt_baseline = create_network(optimizer_type='adam', learning_rate=0.001)
    hist_baseline = train_network(
        model_baseline,
        opt_baseline,
        X_train_subset,
        y_train_subset,
        X_val,
        y_val,
        num_epochs=num_epochs,
    )
    acc_baseline = np.mean(predict(model_baseline.network, X_test) == y_test)
    print(f"Test Accuracy: {acc_baseline*100:.2f}%")

    # 2. L1 Regularization
    print("\n2. L1 Regularization (lambda=0.0001)")
    model_l1, opt_l1 = create_network(optimizer_type='adam', learning_rate=0.001, l1=0.0001)
    hist_l1 = train_network(
        model_l1,
        opt_l1,
        X_train_subset,
        y_train_subset,
        X_val,
        y_val,
        num_epochs=num_epochs,
    )
    acc_l1 = np.mean(predict(model_l1.network, X_test) == y_test)
    print(f"Test Accuracy: {acc_l1*100:.2f}%")

    # 3. L2 Regularization
    print("\n3. L2 Regularization (lambda=0.0001)")
    model_l2, opt_l2 = create_network(optimizer_type='adam', learning_rate=0.001, l2=0.0001)
    hist_l2 = train_network(
        model_l2,
        opt_l2,
        X_train_subset,
        y_train_subset,
        X_val,
        y_val,
        num_epochs=num_epochs,
    )
    acc_l2 = np.mean(predict(model_l2.network, X_test) == y_test)
    print(f"Test Accuracy: {acc_l2*100:.2f}%")

    # 4. Dropout
    print("\n4. Dropout (rate=0.5)")
    model_dropout, opt_dropout = create_network(optimizer_type='adam', learning_rate=0.001,
                                               use_dropout=True, dropout_rate=0.5)
    hist_dropout = train_network(
        model_dropout,
        opt_dropout,
        X_train_subset,
        y_train_subset,
        X_val,
        y_val,
        num_epochs=num_epochs,
    )
    acc_dropout = np.mean(predict(model_dropout.network, X_test) == y_test)
    print(f"Test Accuracy: {acc_dropout*100:.2f}%")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline:       {acc_baseline*100:.2f}%")
    print(f"L1:             {acc_l1*100:.2f}%")
    print(f"L2:             {acc_l2*100:.2f}%")
    print(f"Dropout:        {acc_dropout*100:.2f}%")

    # Plot comparison
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(hist_baseline['loss'], label='Baseline')
    plt.plot(hist_l1['loss'], label='L1')
    plt.plot(hist_l2['loss'], label='L2')
    plt.plot(hist_dropout['loss'], label='Dropout')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(hist_baseline['val_loss'], label='Baseline')
    plt.plot(hist_l1['val_loss'], label='L1')
    plt.plot(hist_l2['val_loss'], label='L2')
    plt.plot(hist_dropout['val_loss'], label='Dropout')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot([a*100 for a in hist_baseline['train_accuracy']], label='Baseline')
    plt.plot([a*100 for a in hist_l1['train_accuracy']], label='L1')
    plt.plot([a*100 for a in hist_l2['train_accuracy']], label='L2')
    plt.plot([a*100 for a in hist_dropout['train_accuracy']], label='Dropout')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot([a*100 for a in hist_baseline['val_accuracy']], label='Baseline')
    plt.plot([a*100 for a in hist_l1['val_accuracy']], label='L1')
    plt.plot([a*100 for a in hist_l2['val_accuracy']], label='L2')
    plt.plot([a*100 for a in hist_dropout['val_accuracy']], label='Dropout')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
