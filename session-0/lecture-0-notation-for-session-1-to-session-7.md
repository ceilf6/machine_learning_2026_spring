# Notation for Sessions 1–7

---

## 1. Row-Vector Convention

By default, sessions 1–7 adopt the **row-vector convention** used in modern ML frameworks:

$$
y = x W + b
$$

where:

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $x$ | $\mathbb{R}^{1 \times d_{in}}$ | Input row vector |
| $W$ | $ \mathbb{R}^{d_{in} \times d_{out}}$ | Weight matrix |
| $b$ | $ \mathbb{R}^{1 \times d_{out}}$ | Bias row vector |
| $y$ | $ \mathbb{R}^{1 \times d_{out}}$ | Output row vector |

### Interpretation

- The **data flows through the matrix** from left to right
- The matrix acts as a **projection layer** transforming $d_{in}$ dimensions to $d_{out}$ dimensions
- This is the dominant convention in NumPy, PyTorch, TensorFlow, and sklearn

### Layer-wise Form (Neural Networks)

For layer $l$ in a neural network:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
$$

$$
a^{(l)} = f^{(l)}(z^{(l)})
$$

where:

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $a^{(l-1)}$ | $ \mathbb{R}^{1 \times n_{l-1}}$ | Input to layer $l$ |
| $W^{(l)}$ | $ \mathbb{R}^{n_{l-1} \times n_{l}}$ | Weight matrix at layer $l$ |
| $b^{(l)}$ | $ \mathbb{R}^{1 \times n_{l}}$ | Bias at layer $l$ |
| $z^{(l)}$ | $ \mathbb{R}^{1 \times n_{l}}$ | Pre-activation at layer $l$ |
| $a^{(l)}$ | $ \mathbb{R}^{1 \times n_{l}}$ | Post-activation at layer $l$ |
| $f^{(l)}$ | — | Activation function at layer $l$ |

### Batch Form (Vectorized Computation)

For a batch of $m$ samples stacked as rows:

$$
Z^{(l)} = A^{(l-1)} W^{(l)} + \mathbf{1} b^{(l)}
$$

where:

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $A^{(l-1)}$ | $ \mathbb{R}^{m \times n_{l-1}}$ | Batch input |
| $W^{(l)}$ | $ \mathbb{R}^{n_{l-1} \times n_{l}}$ | Weight matrix |
| $b^{(l)}$ | $ \mathbb{R}^{1 \times n_{l}}$ | Bias row vector |
| $Z^{(l)}$ | $ \mathbb{R}^{m \times n_{l}}$ | Batch pre-activation |
| $\mathbf{1}$ | $ \mathbb{R}^{m \times 1}$ | Column vector of ones |

### Regression and Classification Batch Form

For regression and binary classification with batch size $n$:

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $X$ | $ \mathbb{R}^{n \times d}$ | Batch input matrix (rows are samples) |
| $Y$ | $ \mathbb{R}^{n \times 1}$ | Batch target matrix (rows are sample targets) |
| $\hat{Y}$ | $ \mathbb{R}^{n \times 1}$ | Batch prediction matrix (rows are sample predictions) |

---

## 2. Key Notation Summary

### Scalars and Vectors

| Symbol | Meaning |
|--------|---------|
| $x^{(i)}$ | $i$-th training sample (input) |
| $y^{(i)}$ | $i$-th training label (target) |
| $\hat{y}^{(i)}$ | Model prediction for sample $i$ |
| $n$ | Number of training samples |
| $d$ | Input dimension / number of features |
| $m$ | Batch size |
| $L$ | Number of layers in a neural network |
| $n_l$ | Number of neurons in layer $l$ |
| $f^{(l)}$ | Activation function at layer $l$: $a^{(l)} = f^{(l)}(z^{(l)})$ |

### Parameters

| Symbol | Meaning |
|--------|---------|
| $w$ or $W$ | Weights |
| $b$ | Bias / intercept |
| $\theta$ | Parameter vector (column form) |
| $\eta$ | Learning rate |
| $\lambda$ | Regularization strength |

### Parameter Updates

For gradient descent parameter updates, use the **leftarrow** notation:

$$
g = \frac{\partial \mathcal{L}}{\partial W}, \quad W \leftarrow W - \eta g
$$

This notation clearly indicates an **in-place update** (mutating the parameter) rather than a mathematical equality or assignment. Do not use `:=` or `\gets` for parameter updates.

### Optimization Algorithms

| Symbol | Meaning |
|--------|---------|
| $g$ | Gradient (mini-batch or full-batch depending on context) |
| $v$ (momentum) | **Velocity** (momentum accumulator): $v \leftarrow \beta v + (1-\beta)g$ |
| $m$ (Adam) | **First moment** (Adam): $m \leftarrow \beta_1 m + (1-\beta_1)g$ |
| $v$ (Adam) | **Second moment** (Adam): $v \leftarrow \beta_2 v + (1-\beta_2)g^2$ |
| $\hat{m}, \hat{v}$ (Adam) | Bias-corrected moments in Adam |
| $\beta, \beta_1, \beta_2$ (Adam) | Exponential decay rates for moving averages |

> **Note on SGD terminology:** In this course, **"SGD"** refers to **Mini-batch SGD** (batch size $B$ where $1 < B \ll n$). The theoretical "One-sample SGD" or "Single-sample SGD" ($B=1$) is rarely used in practice.

### Loss and Gradients

| Symbol | Meaning |
|--------|---------|
| $\mathcal{L}$ | Loss function (average over all samples) |
| $\ell^{(i)}$ | Per-example loss for sample $i$ |
| $\frac{\partial \mathcal{L}}{\partial W}$ | Gradient of loss w.r.t. weights |
| $g$ | **Compact gradient notation:** $g = \frac{\partial \mathcal{L}}{\partial W}$ |
| $\mathcal{B}$ | Mini-batch (set of sample indices) |
| $B$ | Mini-batch size ($1 < B \ll n$) |
| $\delta^{(l)}$ | Error signal at layer $l$ |
| $\sigma(\cdot)$ | Sigmoid activation function |
| $\text{ReLU}(z) = \max(0, z)$ | ReLU activation function |

### Dataset

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^n
$$

---

## 3. Convention Reference

| Context | Convention | Example |
|---------|------------|---------|
| **Neural Networks** | Row-vector world ($y = xW + b$) | $z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$ |
| **Logistic Regression** | Row-vector world | $\hat{y} = \sigma(xW + b)$ |
| **Multiple Linear Regression** | Row-vector world | $\hat{y} = xW + b$ |
| **Simple Linear Regression** | Scalar form | $\hat{y} = wx + b$ |

---

## 4. Mathematical Notation Conventions

### Absolute Value

For absolute value, use **double pipes** `||` instead of single pipe `|`:

$$
|x| \text{ is written as } \|x\|
$$

This avoids ambiguity with:
- Set cardinality notation $|S|$
- Determinant notation $|A|$
- Single pipe as delimiter in Markdown tables

---

## 5. Dimension Check Quick Reference

For any matrix multiplication $C = AB$:

$$
C \in \mathbb{R}^{m \times p} \leftarrow A \in \mathbb{R}^{m \times n} \cdot B \in \mathbb{R}^{n \times p}
$$

Inner dimensions must match; outer dimensions give result shape.

### Common Neural Network Dimensions

```
Input:          x      ∈ ℝ^(1 × d_in)
Layer 1:        W^(1)  ∈ ℝ^(d_in × n_1),    b^(1) ∈ ℝ^(1 × n_1)
Hidden 1:       a^(1)  ∈ ℝ^(1 × n_1)
Layer 2:        W^(2)  ∈ ℝ^(n_1 × n_2),     b^(2) ∈ ℝ^(1 × n_2)
Hidden 2:       a^(2)  ∈ ℝ^(1 × n_2)
Output Layer:   W^(L)  ∈ ℝ^(n_(L-1) × d_out), b^(L) ∈ ℝ^(1 × d_out)
Output:         ŷ      ∈ ℝ^(1 × d_out)
```
