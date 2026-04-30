# Design and Implementation of a Minimal Neural Network for MNIST

This document explains the **design philosophy**, **network structure**, and **backpropagation implementation** of the neural network defined in `my_nn.py`. The goal of the code is to demonstrate how a neural network can be implemented **from scratch using only NumPy**, without relying on frameworks such as PyTorch or TensorFlow.

The design prioritizes:

* Conceptual clarity
* Explicit implementation of forward and backward passes
* Modular layers
* Minimal abstraction

The resulting system makes the **core mechanics of neural networks transparent**.

---

## 1. Design Philosophy

The central design philosophy of this codebase is:

> A neural network is simply a sequence of differentiable transformations applied to data.

Each transformation is implemented as a **Layer object** with two methods:

* `forward(input)` — computes the layer's output given input
* `backward(grad_output)` — computes gradients and propagates them backward

Thus the neural network is represented as a **function composition**:

$$
f(x) = L^{(n)} \circ L^{(n-1)} \circ \cdots \circ L^{(2)} \circ L^{(1)}(x)
$$

Where each $L^{(i)}$ is a layer transformation.

This architecture follows the same conceptual model used by modern deep learning frameworks:

* **PyTorch `nn.Module`**
* **TensorFlow Layers**
* **Keras Sequential models**

But here it is implemented manually to make every step visible and understandable.

---

## 2. Network Representation

The network is represented as a **Python list of Layer objects**.

```python
network = [
    Dense(784, 64),   # Layer 0: First transformation
    ReLU(),           # Layer 1: First activation
    Dense(64, 32),    # Layer 2: Second transformation
    ReLU(),           # Layer 3: Second activation
    Dense(32, 10),    # Layer 4: Output transformation
]
```

This simple list structure means the network is a **linear computational pipeline**:

```
Input → Dense → ReLU → Dense → ReLU → Dense → Output
```

No graph structure, no dynamic computation graph — just a **sequential chain of operations**.

The forward computation is implemented as:

```python
for layer in network:
    input = layer.forward(input)
```

### Mathematical Notation

Let's establish clear notation for the forward pass:

- $x^{(0)}$ — input data (batch of 784-dimensional vectors)
- $z^{(l)}$ — **pre-activation** (output of Dense layer before activation)
- $a^{(l)}$ — **activation** (output after applying activation function)
- $W^{(l)}$ — weight matrix of layer $l$
- $b^{(l)}$ — bias vector of layer $l$

The forward pass computes:

$$
\begin{align}
z^{(1)} &= a^{(0)} W^{(1)} + b^{(1)} \quad &\text{(Dense layer 1)} \\
a^{(1)} &= \text{ReLU}(z^{(1)}) \quad &\text{(Activation 1)} \\
z^{(2)} &= a^{(1)} W^{(2)} + b^{(2)} \quad &\text{(Dense layer 2)} \\
a^{(2)} &= \text{ReLU}(z^{(2)}) \quad &\text{(Activation 2)} \\
z^{(3)} &= a^{(2)} W^{(3)} + b^{(3)} \quad &\text{(Dense layer 3, logits)}
\end{align}
$$

Where $a^{(0)} = x^{(0)}$ is the input, and $z^{(3)}$ are the final **logits** (pre-softmax scores).

---

## 3. Understanding "Activations" in the Code

The term **"activations"** in this codebase refers to **all intermediate outputs** produced during the forward pass, stored in a list for later use.

```python
def forward(network, X):
    activations = []
    input = X
    for layer in network:
        input = layer.forward(input)
        activations.append(input)  # Store output of this layer
    return activations
```

### What Gets Stored

The `activations` list contains:

```python
activations[0] = output of Dense(784, 64)     = z^(1)
activations[1] = output of ReLU()             = a^(1)
activations[2] = output of Dense(64, 32)      = z^(2)
activations[3] = output of ReLU()             = a^(2)
activations[4] = output of Dense(32, 10)      = z^(3) (logits)
```

### Why Store Activations?

1. **For predictions**: `activations[-1]` gives the final logits
2. **For backpropagation**: Some layers need their forward pass outputs to compute gradients (though in this implementation, layers store their own inputs internally)
3. **For debugging/visualization**: Can inspect intermediate representations

### Terminology Clarification

In deep learning literature:

- **Pre-activation**: Output of linear transformation (e.g., $z = xW + b$)
- **Activation**: Output after applying activation function (e.g., $a = \text{ReLU}(z)$)

In this code, `activations` stores **both** pre-activations and activations sequentially.

---

## 4. Network Architecture Details

### Layer Count and Structure

The network contains **5 layers total**:

| Index | Layer Type | Input Dim | Output Dim | Parameters? |
|-------|-----------|-----------|------------|-------------|
| 0 | Dense | 784 | 64 | ✓ Yes |
| 1 | ReLU | 64 | 64 | ✗ No |
| 2 | Dense | 64 | 32 | ✓ Yes |
| 3 | ReLU | 32 | 32 | ✗ No |
| 4 | Dense | 32 | 10 | ✓ Yes |

**3 parameterized layers** (Dense layers with weights and biases)
**2 non-parameterized layers** (ReLU activations)

### Parameter Count

Total trainable parameters:

$$
\begin{align}
\text{Layer 0:} \quad &784 \times 64 + 64 = 50,240 \\
\text{Layer 2:} \quad &64 \times 32 + 32 = 2,080 \\
\text{Layer 4:} \quad &32 \times 10 + 10 = 330 \\
\hline
\text{Total:} \quad &52,650 \text{ parameters}
\end{align}
$$

---

## 5. Dense Layer: Affine Transformation

A Dense (fully connected) layer performs an **affine transformation**:

$$
y = x W + b
$$

Where:
- $x \in \mathbb{R}^{B \times n_{in}}$ — input batch ($B$ = batch size)
- $W \in \mathbb{R}^{n_{in} \times n_{out}}$ — weight matrix
- $b \in \mathbb{R}^{n_{out}}$ — bias vector
- $y \in \mathbb{R}^{B \times n_{out}}$ — output batch

### Code Implementation

```python
return np.dot(input, self.weights) + self.biases
```

The bias is **broadcasted** across the batch dimension automatically by NumPy.

### Why "Fully Connected"?

Each output neuron $j$ depends on **every input neuron** $i$:

$$
y_j = \sum_{i=1}^{n_{in}} x_i W_{ij} + b_j
$$

This creates $n_{in} \times n_{out}$ connections, hence "fully connected."

**Contrast with convolutional layers**: where each output depends on only a local receptive field.

### Connection Pattern Between Layers

When we connect `Dense(64, 32)` after `ReLU()`:

```
[64 neurons] → [32 neurons]
```

Each of the 32 output neurons receives input from **all 64 input neurons** via the weight matrix $W \in \mathbb{R}^{64 \times 32}$.

---

## 6. ReLU Activation Function

The **Rectified Linear Unit (ReLU)** is defined as:

$$
\text{ReLU}(z) = \max(0, z)
$$

Applied element-wise to the input.

### Forward Pass

```python
return np.maximum(0, input)
```

Effect:
- Positive values pass through unchanged
- Negative values become zero
- Introduces non-linearity (essential for learning complex patterns)

### Backward Pass (Gradient)

The derivative is:

$$
\frac{\partial \text{ReLU}(z)}{\partial z} = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

Code implementation:

```python
relu_grad = self.input > 0  # Boolean mask: True where input > 0
return grad_output * relu_grad  # Element-wise multiplication
```

### Why Store `self.input`?

During forward pass:
```python
self.input = input  # Store for backward pass
```

This is necessary because the gradient depends on **which inputs were positive**.

The backward pass needs to know: "For this output gradient, which inputs should I pass it through?"

---

## 7. Why No Softmax Layer in the Network

The network ends with:

```python
Dense(32, 10)  # Output layer producing logits
```

Notably, there is **no Softmax layer** in the `network` list.

### Design Rationale

#### Reason 1: Numerical Stability

Computing softmax then cross-entropy separately:

$$
\begin{align}
p &= \text{softmax}(z) = \frac{e^{z}}{\sum_j e^{z_j}} \\
\mathcal{L} &= -\sum_i y_i \log(p_i)
\end{align}
$$

Can cause numerical overflow when $e^{z_j}$ is large.

**Instead**, the combined function `softmax_crossentropy_with_logits` computes both simultaneously with the **log-sum-exp trick**:

```python
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
```

Subtracting the max prevents overflow while maintaining mathematical equivalence (softmax is translation-invariant).

#### Reason 2: Simplified Gradient

The gradient of cross-entropy loss with respect to logits simplifies beautifully:

$$
\frac{\partial \mathcal{L}}{\partial z} = p - y
$$

Where:
- $p$ = predicted probabilities (after softmax)
- $y$ = true labels (one-hot encoded)

This simple form is directly returned:

```python
grad = (softmax_probs - one_hot_labels) / batch_size
```

If we had separate Softmax and CrossEntropy layers, we'd need to implement both their gradients separately, which is more complex and prone to numerical errors.

---

## 8. Forward Pass Implementation

Forward propagation is implemented in the `forward()` function:

```python
def forward(network, X):
    activations = []
    input = X
    for layer in network:
        input = layer.forward(input)
        activations.append(input)
    return activations
```

### Execution Flow

Given input batch $X \in \mathbb{R}^{B \times 784}$:

1. **Layer 0 (Dense)**: $z^{(1)} = X W^{(1)} + b^{(1)}$ → shape $(B, 64)$
2. **Layer 1 (ReLU)**: $a^{(1)} = \max(0, z^{(1)})$ → shape $(B, 64)$
3. **Layer 2 (Dense)**: $z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}$ → shape $(B, 32)$
4. **Layer 3 (ReLU)**: $a^{(2)} = \max(0, z^{(2)})$ → shape $(B, 32)$
5. **Layer 4 (Dense)**: $z^{(3)} = a^{(2)} W^{(3)} + b^{(3)}$ → shape $(B, 10)$

The final output $z^{(3)}$ contains **logits** (unnormalized scores) for each of the 10 digit classes.

### Why Store All Activations?

While each layer already stores its input internally (e.g., `self.input` in Dense and ReLU), the `activations` list provides:

1. **Convenient access to final output**: `activations[-1]` gives logits
2. **Potential for analysis**: Can inspect intermediate representations
3. **Framework design pattern**: Mirrors how automatic differentiation systems track computation graphs

---

## 9. Backpropagation: The Chain Rule in Action

**Backpropagation** is the algorithm for computing gradients of the loss with respect to all parameters using the **chain rule of calculus**.

### The Chain Rule

For a composite function $f(g(x))$:

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
$$

In a neural network with layers $L^{(1)}, L^{(2)}, \ldots, L^{(n)}$:

$$
\mathcal{L} = f(L^{(n)}(\cdots L^{(2)}(L^{(1)}(x))))
$$

To find $\frac{\partial \mathcal{L}}{\partial W^{(i)}}$ for layer $i$, we need:

$$
\frac{\partial \mathcal{L}}{\partial W^{(i)}} = \frac{\partial \mathcal{L}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial W^{(i)}}
$$

Where $\frac{\partial \mathcal{L}}{\partial z^{(i)}}$ is computed by propagating gradients backward from the loss.

---

## 10. Backpropagation Implementation in Code

Backpropagation is implemented in the `train()` function:

```python
def train(network, X, y):
    # Forward pass
    activations = forward(network, X)
    logits = activations[-1]
    
    # Compute loss and gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)
    
    # Backward pass
    grad_output = grad_logits
    for i in range(len(network))[::-1]:  # Reverse order!
        layer = network[i]
        grad_output = layer.backward(grad_output)
    
    return loss
```

### Step-by-Step Breakdown

#### Step 1: Forward Pass

Compute all layer outputs and store activations.

#### Step 2: Compute Loss and Initial Gradient

```python
loss, grad_logits = softmax_crossentropy_with_logits(logits, y)
```

This returns:
- `loss`: scalar value of cross-entropy loss
- `grad_logits`: $\frac{\partial \mathcal{L}}{\partial z^{(3)}}$ with shape $(B, 10)$

This is the **starting point** for backpropagation.

#### Step 3: Propagate Gradients Backward

```python
for i in range(len(network))[::-1]:  # [4, 3, 2, 1, 0]
    layer = network[i]
    grad_output = layer.backward(grad_output)
```

**Reverse iteration** is crucial! We process layers from output to input:

```
Layer 4 (Dense) → Layer 3 (ReLU) → Layer 2 (Dense) → Layer 1 (ReLU) → Layer 0 (Dense)
```

Each `layer.backward(grad_output)` does two things:

1. **Computes local gradients** (if layer has parameters, updates them)
2. **Returns gradient w.r.t. input** (to pass to previous layer)

### Gradient Flow Notation

Let $\frac{\partial \mathcal{L}}{\partial z^{(l)}}$ denote the gradient flowing backward into layer $l$.

The backward pass computes:

$$
\begin{align}
\text{At Layer 4:} \quad &\frac{\partial \mathcal{L}}{\partial z^{(3)}} \text{ (given from loss)} \\
&\frac{\partial \mathcal{L}}{\partial W^{(3)}}, \frac{\partial \mathcal{L}}{\partial b^{(3)}} \text{ (computed and used for updates)} \\
&\frac{\partial \mathcal{L}}{\partial a^{(2)}} \text{ (returned)} \\
\\
\text{At Layer 3:} \quad &\frac{\partial \mathcal{L}}{\partial a^{(2)}} \text{ (received from Layer 4)} \\
&\frac{\partial \mathcal{L}}{\partial z^{(2)}} \text{ (returned)} \\
\\
\text{At Layer 2:} \quad &\frac{\partial \mathcal{L}}{\partial z^{(2)}} \text{ (received from Layer 3)} \\
&\frac{\partial \mathcal{L}}{\partial W^{(2)}}, \frac{\partial \mathcal{L}}{\partial b^{(2)}} \text{ (computed and used)} \\
&\frac{\partial \mathcal{L}}{\partial a^{(1)}} \text{ (returned)}
\end{align}
$$

And so on...

---

## 11. Dense Layer Backpropagation in Detail

The Dense layer's `backward()` method computes **three gradients**:

```python
def backward(self, grad_output):
    grad_weights = np.dot(self.input.T, grad_output)
    grad_biases = np.sum(grad_output, axis=0)
    grad_input = np.dot(grad_output, self.weights.T)
    
    self.weights = self.weights - self.learning_rate * grad_weights
    self.biases = self.biases - self.learning_rate * grad_biases
    
    return grad_input
```

### Derivation

Given forward pass: $y = x W + b$

And incoming gradient: $\frac{\partial \mathcal{L}}{\partial y}$ (denoted `grad_output`)

We need to compute:

#### 1. Gradient w.r.t. Weights

$$
\frac{\partial \mathcal{L}}{\partial W} = x^T \frac{\partial \mathcal{L}}{\partial y}
$$

**Dimensions**:
- $x$: $(B, n_{in})$
- $\frac{\partial \mathcal{L}}{\partial y}$: $(B, n_{out})$
- $x^T \frac{\partial \mathcal{L}}{\partial y}$: $(n_{in}, B) \times (B, n_{out}) = (n_{in}, n_{out})$ ✓

**Code**:
```python
grad_weights = np.dot(self.input.T, grad_output)
```

**Intuition**: Each weight $W_{ij}$ affects the output through $x_i$, so its gradient accumulates contributions from all samples in the batch.

#### 2. Gradient w.r.t. Biases

$$
\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial y_i}
$$

**Why sum over batch?** Each bias $b_j$ affects **all samples** in the batch equally, so we sum their gradient contributions.

**Code**:
```python
grad_biases = np.sum(grad_output, axis=0)
```

Summing over `axis=0` (batch dimension) gives a vector of shape $(n_{out},)$.

#### 3. Gradient w.r.t. Input

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} W^T
$$

**Dimensions**:
- $\frac{\partial \mathcal{L}}{\partial y}$: $(B, n_{out})$
- $W^T$: $(n_{out}, n_{in})$
- Product: $(B, n_{out}) \times (n_{out}, n_{in}) = (B, n_{in})$ ✓

**Code**:
```python
grad_input = np.dot(grad_output, self.weights.T)
```

**Purpose**: This gradient is **passed to the previous layer** to continue backpropagation.

### Parameter Update (Gradient Descent)

After computing gradients, parameters are updated:

$$
\begin{align}
W &\leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W} \\
b &\leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
\end{align}
$$

Where $\eta$ is the learning rate.

**Code**:
```python
self.weights = self.weights - self.learning_rate * grad_weights
self.biases = self.biases - self.learning_rate * grad_biases
```

---

## 12. ReLU Backpropagation

The ReLU backward pass is simpler:

```python
def backward(self, grad_output):
    relu_grad = self.input > 0
    return grad_output * relu_grad
```

### Derivation

Given forward pass: $a = \max(0, z)$

The element-wise derivative is:

$$
\frac{\partial a_i}{\partial z_i} = \begin{cases}
1 & \text{if } z_i > 0 \\
0 & \text{if } z_i \leq 0
\end{cases}
$$

By chain rule:

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a} \odot \frac{\partial a}{\partial z}
$$

Where $\odot$ denotes element-wise multiplication.

**Code breakdown**:
```python
relu_grad = self.input > 0  # Binary mask: True where z > 0
return grad_output * relu_grad  # Element-wise multiply
```

**Effect**: Gradients **pass through unchanged** where the input was positive, and are **zeroed** where the input was non-positive.

This is why ReLU can cause **dying neurons** — if a neuron always outputs 0, its gradient is always 0, and it never updates.

---

## 13. He Initialization

Weights are initialized using **He initialization**:

```python
self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
```

### Mathematical Justification

For a layer with $n_{in}$ inputs and ReLU activation:

If weights are initialized with variance:

$$
\text{Var}(W) = \frac{2}{n_{in}}
$$

Then the variance of activations is preserved across layers, preventing **vanishing or exploding gradients**.

**Derivation sketch**:
- Input variance: $\text{Var}(x) = 1$ (after normalization)
- Pre-activation: $z = \sum_{i=1}^{n} w_i x_i$
- $\text{Var}(z) = n \cdot \text{Var}(w) \cdot \text{Var}(x) = n \cdot \text{Var}(w)$
- After ReLU (which zeros half the values on average): $\text{Var}(a) \approx \frac{1}{2} \text{Var}(z)$
- Setting $\text{Var}(w) = \frac{2}{n}$ gives $\text{Var}(a) \approx 1$

**Why not Xavier/Glorot initialization?**

Xavier initialization uses $\text{Var}(W) = \frac{1}{n_{in}}$ and is designed for **tanh/sigmoid** activations. He initialization's factor of 2 accounts for ReLU zeroing half its inputs.

---

## 14. Gradient Descent Strategy: Full Batch vs. Mini-Batch

This implementation uses **full-batch gradient descent**:

```python
loss = train(network, X_train, y_train)  # Entire dataset
```

### Full-Batch Gradient Descent

**Each iteration**:
1. Compute forward pass on **entire training set**
2. Compute **exact gradient** of loss
3. Update parameters once

**Advantages**:
- Stable, smooth convergence
- True gradient (not an estimate)
- Simple to implement

**Disadvantages**:
- Slow for large datasets (all data in memory)
- No stochasticity (can get stuck in sharp minima)
- Inefficient use of computation (redundancy in similar samples)

### Comparison to Mini-Batch SGD

**Stochastic Gradient Descent (SGD)** would use small batches (e.g., 32-256 samples):

```python
for batch_X, batch_y in get_batches(X_train, y_train, batch_size=64):
    loss = train(network, batch_X, batch_y)
```

**Advantages of mini-batch**:
- Faster iterations
- Better generalization (noise acts as regularization)
- Can handle datasets larger than memory
- Momentum-based methods work better with noise

**Why full-batch here?**
- MNIST is small (~60k samples, 784 dims)
- Educational clarity
- Demonstrates the "pure" gradient descent algorithm

---

## 15. Prediction Pipeline

Prediction involves three steps:

```python
def predict(network, X):
    logits = forward(network, X)[-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)
```

### Step 1: Compute Logits

Forward pass through the network produces unnormalized scores:

$$
z = f_{\text{network}}(X) \in \mathbb{R}^{B \times 10}
$$

### Step 2: Convert to Probabilities

Apply softmax:

$$
p_j = \frac{e^{z_j}}{\sum_{k=1}^{10} e^{z_k}}
$$

This transforms logits into a valid probability distribution:
- $p_j \in (0, 1)$
- $\sum_j p_j = 1$

### Step 3: Select Most Likely Class

$$
\hat{y} = \argmax_j p_j
$$

Returns the class index (0-9) with highest probability.

**Note**: We could skip softmax and use `np.argmax(logits)` directly since $\argmax$ is invariant to monotonic transformations. But computing probabilities is standard practice and useful for interpreting confidence.

---

## 16. Interesting Implementation Details

### 1. The Network is Framework-Like

This ~200-line implementation replicates the core functionality of:

```python
# PyTorch equivalent
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)
```

But with **explicit forward and backward passes** that reveal the underlying mathematics.

### 2. Layers are Fully Modular

The `Layer` base class defines a common interface:

```python
class Layer:
    def forward(self, input): ...
    def backward(self, grad_output): ...
```

This makes adding new layer types trivial. Examples:

**Sigmoid activation**:
```python
class Sigmoid(Layer):
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, grad_output):
        sigmoid_grad = self.output * (1 - self.output)
        return grad_output * sigmoid_grad
```

**Dropout (can be added later on)**:
```python
class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
    
    def forward(self, input):
        self.mask = np.random.rand(*input.shape) > self.p
        return input * self.mask / (1 - self.p)
    
    def backward(self, grad_output):
        return grad_output * self.mask / (1 - self.p)
```

### 3. Backpropagation is Decentralized

Each layer is **responsible for its own gradient computation**. The network merely orchestrates the **order of execution**.

This mirrors how automatic differentiation systems work:
- Each operation registers a "backward function"
- The autograd engine calls them in reverse order
- No central gradient computation logic

### 4. Everything Reduces to Matrix Operations

Despite the conceptual complexity of "neural networks," all computation is:

- **Matrix multiplication**: `np.dot()` for linear transformations
- **Element-wise operations**: `*` (Hadamard product) for activations and gradients
- **Broadcasting**: Implicit expansion of dimensions for bias addition

This is why neural networks are so efficient on GPUs — they are essentially **dense linear algebra** operations.

---

## 17. NumPy Operations Deep Dive

### 17.1 Matrix Multiplication: `np.dot()`

The workhorse of neural networks is matrix multiplication. In the Dense layer:

```python
# Forward pass
output = np.dot(input, self.weights)  # (B, n_in) × (n_in, n_out) → (B, n_out)

# Backward pass - gradient w.r.t. weights
grad_weights = np.dot(self.input.T, grad_output)  # (n_in, B) × (B, n_out) → (n_in, n_out)

# Backward pass - gradient w.r.t. input
grad_input = np.dot(grad_output, self.weights.T)  # (B, n_out) × (n_out, n_in) → (B, n_in)
```

**Key insight**: Matrix multiplication elegantly handles the "fully connected" nature of Dense layers, computing all outputs for all samples in parallel.

### 17.2 Element-wise Multiplication: The `*` Operator (Hadamard Product)

NumPy's `*` operator performs **element-wise multiplication** (also called the Hadamard product), distinct from matrix multiplication.

#### Usage in the Code

**ReLU backward pass:**
```python
relu_grad = self.input > 0       # Boolean mask: True where input > 0
return grad_output * relu_grad   # Element-wise multiply
```

**Softmax cross-entropy gradient:**
```python
grad = (softmax_probs - one_hot_labels) / batch_size  # Element-wise subtraction
```

**Loss computation:**
```python
loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
```

#### Mathematical Definition

For two matrices $A, B \in \mathbb{R}^{m \times n}$:

$$
(A \odot B)_{ij} = A_{ij} \cdot B_{ij}
$$

Where $\odot$ denotes the Hadamard product.

**Example:**
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
C = A * B
# Result: [[ 5, 12],
#          [21, 32]]

# Matrix multiplication
D = np.dot(A, B)
# Result: [[19, 22],
#          [43, 50]]
```

#### Why Both Operations?

| Operation | Symbol | Use Case |
|-----------|--------|----------|
| Matrix Multiplication | `np.dot()` | Linear transformations, weight application |
| Element-wise (Hadamard) | `*` | Activation functions, gradient masking, loss computation |


### 17.3 Broadcasting: Automatic Dimension Expansion

**Broadcasting** is NumPy's mechanism for performing arithmetic operations on arrays with different shapes.

#### Broadcasting in Dense Layer

```python
return np.dot(input, self.weights) + self.biases
```

Here:
- `np.dot(input, self.weights)` produces output of shape $(B, n_{out})$
- `self.biases` has shape $(n_{out},)$ — a 1D vector

NumPy automatically "broadcasts" the bias vector across the batch dimension.

#### How Broadcasting Works

**Rule**: Two dimensions are compatible when:
1. They are equal, OR
2. One of them is 1

**In our case:**


The bias is conceptually expanded to shape $(1, n_{out})$, then replicated $B$ times along the first axis.

#### Visual Representation

```
Without broadcasting (manual expansion):
  [[z11, z12, ..., z1n]]       [[b1, b2, ..., bn]]
  [[z21, z22, ..., z2n]]   +   [[b1, b2, ..., bn]]
  ...                          ...
  [[zB1, zB2, ..., zBn]]       [[b1, b2, ..., bn]]

With broadcasting (automatic):
  [[z11, z12, ..., z1n]]       [b1, b2, ..., bn]
  [[z21, z22, ..., z2n]]   +   ↑
  ...                          automatically expanded
  [[zB1, zB2, ..., zBn]]
```

#### Why Broadcasting Matters

1. **Memory efficiency**: No actual copying of data occurs
2. **Code clarity**: Write mathematical expressions naturally
3. **Performance**: Vectorized operations are faster than Python loops

#### Broadcasting in Backpropagation

**Bias gradient:**
```python
grad_biases = np.sum(grad_output, axis=0)
```

The reverse operation — summing over the batch dimension — collapses the $(B, n_{out})$ gradient into a $(n_{out},)$ vector matching the bias shape.

**The symmetry is elegant:**
- Forward: broadcasting adds bias to each sample
- Backward: summing aggregates gradients from all samples

---

## 18. Data Loading and Preprocessing

### MNIST Data Format

The `load_mnist_from_csv()` function handles data loading:

```python
def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
```

**Expected CSV format:**
- First column: label (0-9)
- Remaining 784 columns: pixel values (28×28 = 784)

### Normalization

```python
X_train_full = train_data.iloc[:, 1:].to_numpy(np.float32) / 255.0
```

**Why divide by 255?**
- Raw pixel values are integers in range [0, 255]
- Normalization scales them to [0, 1]
- Benefits:
  1. **Numerical stability**: Prevents large activations
  2. **Faster convergence**: Gradients are well-scaled
  3. **Consistent initialization**: He initialization assumes normalized inputs

### Train/Validation Split

```python
n_val = int(len(X_train_full) * val_split)  # 10% for validation
np.random.seed(42)
val_indices = np.random.choice(len(X_train_full), n_val, replace=False)
```

**Purpose of validation set:**
- Monitor for overfitting during training
- Tune hyperparameters (learning rate, architecture)
- Not used for gradient computation


---

## 19. The Complete Training Loop

Putting it all together, the `train_mnist_network()` function orchestrates the entire process:

```python
def train_mnist_network(X_train, y_train, X_val, y_val, num_epochs=200):
    # 1. Initialize network
    network = [
        Dense(784, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 10),
    ]
    
    # 2. Training loop
    for epoch in range(num_epochs):
        # Forward + Backward + Update (all in train())
        loss = train(network, X_train, y_train)
        
        # 3. Evaluation
        train_accuracy = np.mean(predict(network, X_train) == y_train)
        val_accuracy = np.mean(predict(network, X_val) == y_val)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")
```

### What Happens Each Epoch?

```
Epoch Start
    ↓
[Forward Pass]  → Compute all activations
    ↓
[Loss Computation]  → Cross-entropy loss + initial gradient
    ↓
[Backward Pass]  → Propagate gradients, update weights
    ↓
[Evaluation]  → Compute accuracy on train and validation sets
    ↓
Epoch End (repeat)
```

---

## 20. Summary


### NumPy Operations Mastered

1. **`np.dot()`** — Matrix multiplication for linear layers
2. **`*` (Hadamard)** — Element-wise multiplication for activations/gradients
3. **Broadcasting** — Automatic dimension expansion for bias
4. **`np.sum()`** — Aggregation for bias gradients
5. **`np.maximum()`** — Element-wise ReLU activation

### Design Principles

1. **Modularity**: Each layer is self-contained with `forward()` and `backward()`
2. **Explicitness**: No hidden magic — all gradients computed manually
3. **Efficiency**: Vectorized NumPy operations over Python loops
4. **Clarity**: ~200 lines that reveal the essence of neural networks
