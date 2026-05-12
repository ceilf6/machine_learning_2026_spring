# Backpropagation in Neural Networks

![](./img/trainnncover.gif)

---

## 1. Neural Network Structure

Consider a feedforward neural network with $L$ layers. For layer $l$:

* Input: $a^{(l-1)} \in \mathbb{R}^{1 \times n_{l-1}}$
* Weights: $W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}$
* Biases: $b^{(l)} \in \mathbb{R}^{1 \times n_l}$
* Pre-activation: $z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$
* Activation: $a^{(l)} = f_l(z^{(l)})$

The output is $a^{(L)}$, and the network loss is $\mathcal{L}(a^{(L)}, y)$, where $y$ is the target.

Our goal is to compute:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}}, \quad l = 1, \dots, L
$$

---

## 2. Forward Pass

The forward pass computes all activations and stores intermediate variables:

$$
\begin{aligned}
z^{(1)} &= x W^{(1)} + b^{(1)}, & a^{(1)} &= f_1(z^{(1)}) \\
z^{(2)} &= a^{(1)} W^{(2)} + b^{(2)}, & a^{(2)} &= f_2(z^{(2)}) \\
& \vdots & & \vdots \\
z^{(L)} &= a^{(L-1)} W^{(L)} + b^{(L)}, & a^{(L)} &= f_L(z^{(L)})
\end{aligned}
$$

These stored $z^{(l)}$ and $a^{(l)}$ are necessary for backpropagation.

---

## 3. Error Signals and Chain Rule

Backpropagation computes an **error signal** $\delta^{(l)}$ for each layer:

$$
\boxed{\underbrace{\delta^{(l)}}_{\text{error signal}} = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{(l)}}}_{\text{gradient w.r.t. pre-activation}} \in \mathbb{R}^{1 \times n_l}}
$$

> [!INFO]
> **$\delta$ vs. $g$ (both are gradients, but w.r.t. different variables)**
>
> In these notes:
>
> - **$\delta^{(l)}$** is the *backprop error signal* at layer $l$:
>   $$
>   \delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}
>   $$
>
>   It is a gradient w.r.t. an **intermediate quantity** (the pre-activation $z^{(l)}$), used to propagate information backward efficiently.
>
> - **$g$** is usually reserved for the *parameter gradient* used by the optimizer update, e.g.
>   $$
>   g^{(l)} = \frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad W^{(l)} \leftarrow W^{(l)} - \eta\, g^{(l)}
>   $$
>
> They are linked by chain rule. Once we have $\delta^{(l)}$, we can compute the parameter gradients:
> $$
> \frac{\partial \mathcal{L}}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
> $$

The **main task** is to compute $\delta^{(l)}$ recursively from the output layer backward. Once $\delta^{(l)}$ is known, the parameter gradients follow directly.

---

## 4. Output Layer Gradients

At the output layer $L$:

$$
\boxed{\underbrace{\delta^{(L)}}_{\text{output error}} = \underbrace{\frac{\partial \mathcal{L}}{\partial a^{(L)}}}_{\text{loss gradient}} \odot \underbrace{f_L'(z^{(L)})}_{\text{activation derivative}}}
$$

Where:

* $\frac{\partial \mathcal{L}}{\partial a^{(L)}}$ depends on the loss function.
* $f_L'(z^{(L)})$ is the derivative of the output activation.
* $\odot$ is element-wise multiplication.

**Examples:**

| Task | Output activation | Loss | Output error $\delta^{(L)}$ |
| :--- | :--- | :--- | :--- |
| Regression | Linear (none) | MSE: $\mathcal{L} = (a^{(L)} - y)^2$ | $2(a^{(L)} - y)$ |
| Binary classification | Sigmoid | BCE: $\mathcal{L} = -\big(y \log a^{(L)} + (1-y)\log(1-a^{(L)})\big)$ | $a^{(L)} - y$ |
| Multiclass classification | Softmax | CE: $\mathcal{L} = -\sum_{k=1}^{K} y_k \log a^{(L)}_k$ | $a^{(L)} - y$ |

**Remark:** For both classification tasks the output error simplifies exactly to **prediction minus target**. The standard MSE regression error is $2(a^{(L)} - y)$ — the factor $2$ comes from differentiating $(a^{(L)} - y)^2$. The Softmax + Cross-Entropy case is particularly elegant: the Jacobian of softmax and the gradient of CE algebraically cancel so that $\delta^{(L)} = a^{(L)} - y$ element-wise.

---

## 5. Hidden Layer Gradients

For hidden layer $l$:

$$
\boxed{ \underbrace{\delta^{(l)}}_{\text{hidden layer error}} = \underbrace{\delta^{(l+1)}}_{\text{error from next layer}} \underbrace{(W^{(l+1)})^T}_{\text{weight transpose}} \odot \underbrace{f_l'(z^{(l)})}_{\text{activation derivative}}}
$$

Explanation:

1. $\delta^{(l+1)} (W^{(l+1)})^T$ propagates the output error backward.
2. $f_l'(z^{(l)})$ scales the error by the derivative of the activation function.
3. This is **chain rule applied layer by layer**.

---

## 6. Weight and Bias Gradients

Once $\delta^{(l)}$ is known:

$$
\boxed{\underbrace{\frac{\partial \mathcal{L}}{\partial W^{(l)}}}_{\text{weight gradient}} = \underbrace{(a^{(l-1)})^T}_{\text{input transpose}} \, \underbrace{\delta^{(l)}}_{\text{error signal}}}
$$

$$
\boxed{\underbrace{\frac{\partial \mathcal{L}}{\partial b^{(l)}}}_{\text{bias gradient}} = \underbrace{\delta^{(l)}}_{\text{error signal}}}
$$

**Interpretation:**

* Each weight gradient measures **how changing the weight changes the loss**.
* Each bias gradient measures **how changing the bias changes the loss**.

---

## 7. Activation Function Derivatives

![](./img/activation-functions2.jpg)

Common activations and derivatives:

| Activation | Function            | Derivative                       |
| ---------- | ------------------- | -------------------------------- |
| Sigmoid    | $f(z) = \sigma(z)$  | $f'(z) = \sigma(z)(1-\sigma(z))$ |
| ReLU       | $f(z) = \max(0, z)$ | $f'(z) = 1$ if $z>0$, else $0$   |
| Tanh       | $f(z) = \tanh(z)$   | $f'(z) = 1 - \tanh^2(z)$         |

These derivatives are applied element-wise to compute $\delta^{(l)}$.

---

## 8. Example: Neural Network for MNIST Digit Recognition

A concrete illustration: input $x \in \mathbb{R}^{1 \times 784}$ (flattened 28×28 image), hidden layer with ReLU, and 10 output neurons for digits 0–9.

### 8.1 Forward Pass

$$
\begin{gathered}
z^{(1)} = x W^{(1)} + b^{(1)}, \quad a^{(1)} = \text{ReLU}(z^{(1)}) \\
z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}, \quad \hat{y} = \text{softmax}(z^{(2)})
\end{gathered}
$$

Here $W^{(2)} \in \mathbb{R}^{n_1 \times 10}$ and $\hat{y} \in \mathbb{R}^{1 \times 10}$ is the probability vector over digits.

### 8.2 Loss

Cross-entropy with one-hot target $y \in \mathbb{R}^{1 \times 10}$:

$$
\mathcal{L} = -\sum_{k=1}^{10} y_k \log \hat{y}_k
$$

### 8.3 Backward Pass

1. **Output error** (Softmax + CE cancellation):

$$
\delta^{(2)} = \hat{y} - y
$$

2. **Output layer gradients**:

$$
\frac{\partial \mathcal{L}}{\partial W^{(2)}} = (a^{(1)})^T \delta^{(2)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(2)}} = \delta^{(2)}
$$

3. **Hidden layer error**:

$$
\delta^{(1)} = \delta^{(2)} (W^{(2)})^T \odot \mathbb{1}_{z^{(1)} > 0}
$$

4. **Hidden layer gradients**:

$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = x^T \delta^{(1)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(1)}} = \delta^{(1)}
$$

5. **Update**:

$$
W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial b^{(l)}}
$$

---

## 9. Vectorized Backpropagation

For a batch of $m$ samples with inputs $X \in \mathbb{R}^{m \times d}$ (stacked as rows):

$$
\underbrace{\Delta^{(l)}}_{\text{batch error signals}} = \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{(l)}}}_{\text{gradient w.r.t. pre-activations}} \quad \text{(matrix of errors for all samples)}
$$

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{(l)}}}_{\text{weight gradient}} = \underbrace{\frac{1}{m}}_{\text{batch averaging}} \underbrace{(A^{(l-1)})^T}_{\text{input transpose}} \underbrace{\Delta^{(l)}}_{\text{error signals}}
$$

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial b^{(l)}}}_{\text{bias gradient}} = \underbrace{\frac{1}{m}}_{\text{batch averaging}} \sum_{i=1}^{m} \underbrace{\Delta^{(l)}_{i,:}}_{\text{error for sample } i}
$$

Vectorization ensures **efficient computation** on GPUs for deep networks.

