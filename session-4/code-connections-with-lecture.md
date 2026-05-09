# Code Connections: How Math Becomes Python Implementation

This document maps every key mathematical concept from the backpropagation lectures to its concrete implementation in `code-my_nn.py`. Each section shows the theory, the code, and the connection between them.

---

## 1. Forward Pass: From Math to Code

### 1.1 Mathematical Foundation (Lecture 1)

**Mathematical representation:**
$$\hat{y} = f(x; W, b)$$

For a neural network with $L$ layers:
$$a^{(l)} = f_l(z^{(l)}), \quad z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$$

**Code implementation:**
```python
# Dense layer forward pass (lines 71-73)
def forward(self, input):
    self.input = input  # Store for backward pass
    return np.dot(input, self.weights) + self.biases

# ReLU activation forward pass (lines 52-54)
def forward(self, input):
    self.input = input  # Store for backward pass
    return np.maximum(0, input)

# Network forward pass (lines 116-124)
def forward(network, X):
    activations = []
    input = X
    for layer in network:
        input = layer.forward(input)
        activations.append(input)
    return activations
```

**Connection:**
- `np.dot(input, self.weights) + self.biases` implements $z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$
- `np.maximum(0, input)` implements ReLU activation $f_l(z) = \max(0, z)$
- The loop `for layer in network:` implements the sequential computation through all layers

### 1.2 Row-Vector Convention in Practice

**Mathematical notation (from notation guide):**
$$y = xW + b$$
where $x \in \mathbb{R}^{1 \times d_{in}}$, $W \in \mathbb{R}^{d_{in} \times d_{out}}$

**Code implementation:**
```python
# Dense layer uses row-vector convention
return np.dot(input, self.weights) + self.biases
# input.shape: (batch_size, input_units)
# self.weights.shape: (input_units, output_units)
# result.shape: (batch_size, output_units)
```

**Connection:** The code follows the row-vector convention exactly as specified in the notation guide.

---

## 2. Loss Function and Initial Gradient

### 2.1 Mathematical Foundation (Lectures 1, 4)

**For multi-class classification:**
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

**Initial gradient for backprop:**
$$\frac{\partial \mathcal{L}}{\partial z^{(L)}} = \hat{y} - y$$

**Code implementation:**
```python
# Softmax with cross-entropy loss (lines 92-108)
def softmax_crossentropy_with_logits(logits, labels):
    batch_size = logits.shape[0]
    # Create one-hot encoding
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1
    
    # Compute softmax with numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
    
    # Gradient of loss w.r.t. logits
    grad = (softmax_probs - one_hot_labels) / batch_size
    
    return loss, grad
```

**Connection:**
- `softmax_probs - one_hot_labels` implements $\hat{y} - y$
- The division by `batch_size` implements the averaging $\frac{1}{n}$
- `1e-9` provides numerical stability for $\log$ operations

---

## 3. Backpropagation: Chain Rule in Code

### 3.1 Mathematical Foundation (Lectures 2, 3)

**Chain rule for neural networks:**
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

**Error signal propagation:**
$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} = \delta^{(l+1)} (W^{(l+1)})^T \odot f_l'(z^{(l)})$$

### 3.2 Dense Layer Backward Pass

**Mathematical formulas:**
$$\frac{\partial z^{(l)}}{\partial W^{(l)}} = (a^{(l-1)})^T$$

$$\frac{\partial z^{(l)}}{\partial b^{(l)}} = 1$$

$$\frac{\partial z^{(l)}}{\partial a^{(l-1)}} = (W^{(l)})^T$$

**Code implementation:**
```python
# Dense layer backward pass (lines 75-89)
def backward(self, grad_output):
    # Gradient of loss w.r.t. weights: input^T · grad_output
    grad_weights = np.dot(self.input.T, grad_output)
    
    # Gradient of loss w.r.t. biases: sum grad_output over batch dimension
    grad_biases = np.sum(grad_output, axis=0)
    
    # Gradient of loss w.r.t. input: grad_output · weights^T
    grad_input = np.dot(grad_output, self.weights.T)
    
    # Update parameters using gradient descent
    self.weights = self.weights - self.learning_rate * grad_weights
    self.biases = self.biases - self.learning_rate * grad_biases
    
    return grad_input
```

**Connection:**
- `np.dot(self.input.T, grad_output)` implements $(a^{(l-1)})^T \cdot \delta^{(l)}$
- `np.sum(grad_output, axis=0)` implements $\sum_i \delta^{(l)}_i$ for bias gradient
- `np.dot(grad_output, self.weights.T)` implements $\delta^{(l)} \cdot (W^{(l)})^T$
- Parameter updates implement $W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}$

### 3.3 ReLU Activation Backward Pass

**Mathematical derivative:**
$$\frac{\partial \text{ReLU}(z)}{\partial z} = \begin{cases} 
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0 
\end{cases}$$

**Code implementation:**
```python
# ReLU backward pass (lines 56-58)
def backward(self, grad_output):
    relu_grad = self.input > 0  # Boolean mask: True where input > 0
    return grad_output * relu_grad
```

**Connection:**
- `self.input > 0` creates a binary mask implementing the ReLU derivative
- Element-wise multiplication implements the chain rule: $\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a} \odot f'(z)$

---

## 4. Training Loop: Full Backpropagation Algorithm

### 4.1 Mathematical Algorithm (Lecture 6)

**Complete backpropagation steps:**
1. Forward pass: compute $z^{(l)}$ and $a^{(l)}$ for all layers
2. Compute output error: $\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \odot f_L'(z^{(L)})$
3. Backpropagate error: $\delta^{(l)} = \delta^{(l+1)} (W^{(l+1)})^T \odot f_l'(z^{(l)})$
4. Compute gradients and update parameters

### 4.2 Code Implementation

**Training function:**
```python
# Training function (lines 136-151)
def train(network, X, y):
    # Forward pass
    activations = forward(network, X)
    logits = activations[-1]
    
    # Compute loss and initial gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)
    
    # Backward pass (backpropagation)
    grad_output = grad_logits
    for i in range(len(network))[::-1]:  # Reversed order
        layer = network[i]
        grad_output = layer.backward(grad_output)
    
    return loss
```

**Connection:**
- `activations = forward(network, X)` implements step 1: forward pass
- `grad_logits = softmax_crossentropy_with_logits(logits, y)` implements step 2: output error
- `for i in range(len(network))[::-1]` implements step 3: backward pass through layers
- Each `layer.backward(grad_output)` call implements step 4: compute gradients and update

---

## 5. Parameter Initialization

### 5.1 Mathematical Foundation

**He initialization for ReLU networks:**
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

### 5.2 Code Implementation

```python
# Dense layer initialization (lines 65-69)
def __init__(self, input_units, output_units, learning_rate=0.1):
    self.learning_rate = learning_rate
    
    # He initialization: variance proportional to 2/input_units
    self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
    self.biases = np.zeros(output_units)
```

**Connection:**
- `np.random.randn(input_units, output_units)` generates $\mathcal{N}(0, 1)$
- `* np.sqrt(2.0 / input_units)` scales to $\mathcal{N}(0, \sqrt{2/n_{in}})$
- `np.zeros(output_units)` initializes biases to zero

---

## 6. Vectorization and Batch Processing

### 6.1 Mathematical Foundation (Lecture 6)

**Vectorized operations for batch size $m$:**
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{m} (A^{(l-1)})^T \Delta^{(l)}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{m} \sum_{i=1}^{m} \Delta^{(l)}_{i,:}$$

### 6.2 Code Implementation

**Vectorized gradients:**
```python
# In Dense.backward() - already vectorized
grad_weights = np.dot(self.input.T, grad_output)  # (input_units^T · batch_size) · (batch_size · output_units)
grad_biases = np.sum(grad_output, axis=0)         # Sum over batch dimension
```

**Connection:**
- `self.input.T` is $(A^{(l-1)})^T$
- `grad_output` is $\Delta^{(l)}$
- `np.dot(self.input.T, grad_output)` computes $(A^{(l-1)})^T \Delta^{(l)}$
- `np.sum(grad_output, axis=0)` computes $\sum_{i=1}^{m} \Delta^{(l)}_{i,:}$
- The division by batch size happens in `softmax_crossentropy_with_logits`

---

## 7. Specific Model Implementations

### 7.1 Linear Regression Connection

**Mathematical model (Lecture 4):**
$$\hat{y} = xW + b$$
$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2$$
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{2}{n} X^T (\hat{Y} - Y)$$

**How it would appear in code:**
```python
# Linear regression would use a single Dense layer with no activation
# and MSE loss instead of cross-entropy
def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mse_grad(predictions, targets):
    return 2 * (predictions - targets) / len(targets)
```

### 7.2 Logistic Regression Connection

**Mathematical model (Lecture 4):**
$$\hat{y} = \sigma(xW + b)$$
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} [y^{(i)} \log \hat{y}^{(i)} + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} X^T (\hat{Y} - Y)$$

**Current code handles multi-class version:**
```python
# The current implementation is the multi-class extension
# Binary logistic regression would use:
# - Single output neuron instead of 10
# - Sigmoid activation instead of softmax
# - Binary cross-entropy instead of categorical cross-entropy
```

---

## 8. Optimization Algorithm

### 8.1 Mathematical Foundation

**Gradient descent update:**
$$W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}$$

### 8.2 Code Implementation

```python
# In Dense.backward() - lines 85-87
self.weights = self.weights - self.learning_rate * grad_weights
self.biases = self.biases - self.learning_rate * grad_biases
```

**Connection:** Direct implementation of the mathematical update rule.

---

## 9. Network Architecture Definition

### 9.1 Mathematical Architecture

**3-layer network for MNIST:**
- Input: $x \in \mathbb{R}^{784}$ (28×28 flattened image)
- Hidden layer 1: $h_1 \in \mathbb{R}^{64}$ with ReLU
- Hidden layer 2: $h_2 \in \mathbb{R}^{32}$ with ReLU  
- Output: $\hat{y} \in \mathbb{R}^{10}$ with softmax

### 9.2 Code Implementation

```python
# Network definition (lines 163-169)
network = [
    Dense(784, 64),  # Input layer -> Hidden layer 1
    ReLU(),         # Activation function
    Dense(64, 32),  # Hidden layer 1 -> Hidden layer 2
    ReLU(),         # Activation function
    Dense(32, 10),  # Hidden layer 2 -> Output layer
]
```

**Connection:** Each `Dense` layer implements $z = aW + b$, each `ReLU` implements $a = \max(0, z)$.

---

## 10. Prediction and Evaluation

### 10.1 Mathematical Foundation

**Prediction:**
$$\hat{y} = \arg\max(\text{softmax}(z^{(L)}))$$

**Accuracy:**
$$\text{accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}[\hat{y}^{(i)} = y^{(i)}]$$

### 10.2 Code Implementation

```python
# Prediction function (lines 127-133)
def predict(network, X):
    logits = forward(network, X)[-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)

# Accuracy calculation (lines 189-194)
train_predictions = predict(network, X_train)
train_accuracy = np.mean(train_predictions == y_train)
```

**Connection:**
- `softmax(logits)` computes $\text{softmax}(z^{(L)})$
- `np.argmax(probs, axis=-1)` computes $\arg\max$
- `np.mean(predictions == labels)` computes accuracy

---

## 11. Key Design Patterns and Their Mathematical Basis

### 11.1 Storage of Intermediate Values

**Mathematical requirement:** Backpropagation needs forward pass values
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (a^{(l-1)})^T \cdot \delta^{(l)}$$

**Code implementation:**
```python
# Both Dense and ReLU store input during forward pass
def forward(self, input):
    self.input = input  # Store for backward pass
    return computation(input)
```

### 11.2 Modular Layer Design

**Mathematical principle:** Each layer only needs local derivatives
$$\delta^{(l)} = \delta^{(l+1)} (W^{(l+1)})^T \odot f_l'(z^{(l)})$$

**Code implementation:**
```python
# Each layer handles its own backward pass
class Layer:
    def backward(self, grad_output):
        # Each layer implements its local chain rule contribution
        pass
```

### 11.3 Automatic Gradient Flow

**Mathematical chain rule:** Gradients flow backward through the network
$$\frac{\partial \mathcal{L}}{\partial a^{(l-1)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial a^{(l-1)}}$$

**Code implementation:**
```python
# Each layer returns grad_input to pass to previous layer
def backward(self, grad_output):
    # ... compute local gradients ...
    return grad_input  # This becomes grad_output for previous layer
```

---

## 12. Summary of Math-to-Code Mappings

| Mathematical Concept | Code Location | Implementation |
|---------------------|---------------|----------------|
| Forward pass $z = aW + b$ | `Dense.forward()` | `np.dot(input, weights) + biases` |
| ReLU activation $a = \max(0,z)$ | `ReLU.forward()` | `np.maximum(0, input)` |
| Cross-entropy loss | `softmax_crossentropy_with_logits()` | `-np.sum(one_hot * log(softmax)) / batch` |
| Initial gradient $\hat{y} - y$ | `softmax_crossentropy_with_logits()` | `softmax_probs - one_hot_labels` |
| Weight gradient $(a^{(l-1)})^T \delta^{(l)}$ | `Dense.backward()` | `np.dot(self.input.T, grad_output)` |
| Bias gradient $\sum \delta^{(l)}$ | `Dense.backward()` | `np.sum(grad_output, axis=0)` |
| Gradient flow $\delta^{(l)} W^T$ | `Dense.backward()` | `np.dot(grad_output, self.weights.T)` |
| ReLU derivative | `ReLU.backward()` | `grad_output * (self.input > 0)` |
| Parameter update $W \leftarrow W - \eta g$ | `Dense.backward()` | `self.weights - learning_rate * grad_weights` |
| He initialization | `Dense.__init__()` | `np.random.randn() * np.sqrt(2/input_units)` |

This mapping demonstrates how every mathematical formula in the lectures has a direct, line-by-line correspondence in the Python implementation, making the connection between theory and practice explicit and traceable.