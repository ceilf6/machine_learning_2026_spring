# Full Gradient Derivation for Neural Networks

In this lecture, we consolidate everything from the previous lectures and present **all gradients in their final, general form**. This is the culmination of understanding **forward pass → chain rule → backpropagation → parameter updates**.

![](./img/backpropagation_main2.gif)

---

## 1. Network Setup

Consider a feedforward neural network with $L$ layers:

* Input: $a^{(0)} = x \in \mathbb{R}^{1 \times d_0}$
* Pre-activation: $z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$
* Activation: $a^{(l)} = f_l(z^{(l)})$
* Loss: $\mathcal{L}(a^{(L)}, y)$

After the forward pass, all $z^{(l)}$ and $a^{(l)}$ are stored for backpropagation.

---

## 2. Error Signals

Define the error signal at layer $l$:

$$
\underbrace{\delta^{(l)}}_{\text{error signal}} = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{(l)}}}_{\text{gradient w.r.t. pre-activation}} \in \mathbb{R}^{1 \times n_l}
$$

**Output layer:**

$$
\boxed{\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \odot f_L'(z^{(L)})}
$$

**Hidden layers (recursive, $l = L-1, \dots, 1$):**

$$
\boxed{\delta^{(l)} = \delta^{(l+1)} (W^{(l+1)})^T \odot f_l'(z^{(l)})}
$$

---

## 3. Parameter Gradients

Once all $\delta^{(l)}$ are computed:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}}
$$

$$
\boxed{\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}}
$$

---

## 4. Vectorized Form (Batch of $m$ samples)

* Input matrix: $X \in \mathbb{R}^{m \times d_0}$
* Pre-activation: $Z^{(l)} = A^{(l-1)} W^{(l)} + \mathbf{1} b^{(l)}$

**Batch error signals:**

$$
\boxed{\Delta^{(L)} = \frac{\partial \mathcal{L}}{\partial A^{(L)}} \odot f_L'(Z^{(L)})}
$$

$$
\boxed{\Delta^{(l)} = \Delta^{(l+1)} (W^{(l+1)})^T \odot f_l'(Z^{(l)}), \quad l = L-1, \dots, 1}
$$

**Averaged gradients:**

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{m} (A^{(l-1)})^T \Delta^{(l)}}
$$

$$
\boxed{\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{m} \sum_{i=1}^{m} \Delta^{(l)}_{i,:}}
$$


---

## 5. Full Algorithm

1. **Forward pass**: compute and store $z^{(l)}, a^{(l)}$ for all layers
2. **Output error**: $\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \odot f_L'(z^{(L)})$
3. **Backpropagate**: $\delta^{(l)} = \delta^{(l+1)} (W^{(l+1)})^T \odot f_l'(z^{(l)})$
4. **Gradients**: $\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$
5. **Update**: $W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial b^{(l)}}$

---

## 6. Intuition

* Each layer receives **an error signal proportional to its contribution to the final loss**
* Gradients combine **input transpose × error**
* Backpropagation is **just an efficient way to apply the chain rule through all layers**
* Vectorization ensures **fast computation for batches**
