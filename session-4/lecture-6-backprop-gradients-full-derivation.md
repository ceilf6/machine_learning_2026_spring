# Full Gradient Derivation for Neural Networks

In this lecture, we consolidate everything from the previous lectures and present **all gradients in their final, general form**. This is the culmination of understanding **forward pass → chain rule → backpropagation → parameter updates**.


![](./img/backpropagation_main2.gif)

---

## 1. Neural Network Setup

Consider a feedforward neural network with $L$ layers:

* Input: $a^{(0)} = x \in \mathbb{R}^{1 \times d_0}$
* Hidden layers: $a^{(l)} = f_l(z^{(l)})$, $l=1,...,L-1$
* Output: $a^{(L)} = f_L(z^{(L)})$
* Layer pre-activation: $z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$
* Loss function: $\mathcal{L}(a^{(L)}, y)$

Parameters:

$$
\Theta = \{ W^{(1)}, b^{(1)}, ..., W^{(L)}, b^{(L)} \}
$$

Goal:

$$
\text{Compute } \frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}}, \quad \forall l
$$

---

## 2. Forward Pass Recap

Forward pass computes and stores:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}, \quad a^{(l)} = f_l(z^{(l)}), \quad l=1,...,L
$$

These stored $z^{(l)}$ and $a^{(l)}$ are **required for backpropagation**.

---

## 3. Backpropagation: Layer-wise Error Signals

Define **error signal** for each layer:

$$
\underbrace{\delta^{(l)}}_{\text{error signal}} = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{(l)}}}_{\text{gradient w.r.t. pre-activation}} \in \mathbb{R}^{1 \times n_l}
$$

Where $n_l$ is the number of neurons in layer $l$.

### Output Layer:

$$
\underbrace{\delta^{(L)}}_{\text{output error}} = \underbrace{\frac{\partial \mathcal{L}}{\partial a^{(L)}}}_{\text{loss gradient}} \odot \underbrace{f_L'(z^{(L)})}_{\text{activation derivative}}
$$

* $f_L'(z^{(L)})$ is the derivative of the output activation function
* $\odot$ is element-wise multiplication

Examples:

* MSE + linear output:

$$
\delta^{(L)} = a^{(L)} - y
$$

* BCE + sigmoid output:

$$
\delta^{(L)} = a^{(L)} - y
$$

---

### Hidden Layers:

Propagate backward recursively:

$$
\underbrace{\delta^{(l)}}_{\text{hidden layer error}} = \underbrace{\delta^{(l+1)}}_{\text{error from next layer}} \underbrace{(W^{(l+1)})^T}_{\text{weight transpose}} \odot \underbrace{f_l'(z^{(l)})}_{\text{activation derivative}}, \quad \underbrace{l=L-1,...,1}_{\text{backward propagation}}
$$

Interpretation:

* $\delta^{(l+1)} (W^{(l+1)})^T$ transfers the error from the next layer
* $f_l'(z^{(l)})$ adjusts the error based on the local activation function

This is the **chain rule applied through the network**.

---

## 4. Weight and Bias Gradients

Once $\delta^{(l)}$ is known:

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{(l)}}}_{\text{weight gradient}} = \underbrace{(a^{(l-1)})^T}_{\text{input transpose}} \cdot \underbrace{\delta^{(l)}}_{\text{error signal}}
$$

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial b^{(l)}}}_{\text{bias gradient}} = \underbrace{\delta^{(l)}}_{\text{error signal}}
$$

These formulas are **general** for all fully connected layers.

**Intuition:**

* Each weight gradient scales with **input transpose × error signal**
* Each bias gradient scales with **error signal alone**

---

## 5. Vectorized Form (Batch Input)

For a batch of $m$ samples stacked as rows:

* Input matrix: $X \in \mathbb{R}^{m \times d_0}$
* Activation matrix for layer $l$: $A^{(l)} \in \mathbb{R}^{m \times n_l}$
* Pre-activation: $Z^{(l)} = A^{(l-1)} W^{(l)} + \mathbf{1} b^{(l)}$

Error signals:

$$
\underbrace{\Delta^{(L)}}_{\text{batch output error}} = \underbrace{\frac{\partial \mathcal{L}}{\partial A^{(L)}}}_{\text{loss gradient}} \odot \underbrace{f_L'(Z^{(L)})}_{\text{activation derivative}}
$$

Recursive:

$$
\underbrace{\Delta^{(l)}}_{\text{batch hidden error}} = \underbrace{\Delta^{(l+1)}}_{\text{error from next layer}} \underbrace{(W^{(l+1)})^T}_{\text{weight transpose}} \odot \underbrace{f_l'(Z^{(l)})}_{\text{activation derivative}}, \quad \underbrace{l=L-1,...,1}_{\text{backward propagation}}
$$

Weight and bias gradients (averaged over batch):

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial W^{(l)}}}_{\text{weight gradient}} = \underbrace{\frac{1}{m}}_{\text{batch averaging}} \underbrace{(A^{(l-1)})^T}_{\text{input transpose}} \underbrace{\Delta^{(l)}}_{\text{error signals}}
$$

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial b^{(l)}}}_{\text{bias gradient}} = \underbrace{\frac{1}{m}}_{\text{batch averaging}} \sum_{i=1}^{m} \underbrace{\Delta^{(l)}_{i,:}}_{\text{error for sample } i}
$$

This is the **final vectorized form** used in practical neural network libraries.

---

## 6. Special Cases for Common Activations

| Activation | Derivative $f'(z)$       |
| ---------- | ------------------------ |
| Sigmoid    | $\sigma(z)(1-\sigma(z))$ |
| ReLU       | $1$ if $z>0$, else $0$   |
| Tanh       | $1 - \tanh^2(z)$         |
| Linear     | $1$                      |


---

## 7. Summary of Full Backprop Steps

For any neural network:

1. **Forward Pass**: compute $z^{(l)}$ and $a^{(l)}$ for all layers
2. **Compute Output Error**: $\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \odot f_L'(z^{(L)})$
3. **Backpropagate Error**: $\delta^{(l)} = \delta^{(l+1)} (W^{(l+1)})^T \odot f_l'(z^{(l)})$
4. **Compute Gradients**:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
$$

5. **Update Parameters** (Gradient Descent):

$$
W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial b^{(l)}}
$$

---

## 8. Intuition

* Each layer receives **an error signal proportional to its contribution to the final loss**
* Gradients combine **input transpose × error**
* Backpropagation is **just an efficient way to apply the chain rule through all layers**
* Vectorization ensures **fast computation for batches**

---

## 9. Conclusion

The formulas in this lecture represent the **final, complete derivation of gradients** for a fully connected feedforward network. Once implemented, they allow any network to be **trained efficiently using gradient descent** or its variants (SGD, Adam, Momentum).

This lecture **closes the loop** of the backpropagation series:

1. Forward pass → compute activations
2. Loss calculation → measure error
3. Backward pass → compute $\delta^{(l)}$ layer by layer
4. Weight & bias gradients → update parameters
