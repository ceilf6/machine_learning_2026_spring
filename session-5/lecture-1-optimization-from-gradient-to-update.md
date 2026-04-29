# From Gradient to Parameter Update

---

## 1. Review — What We Already Have

By now, after backpropagation, we can compute the gradient of the loss $\mathcal{L}$ with respect to the parameters $W$:

$$
\frac{\partial \mathcal{L}}{\partial W}
$$

This tells us **the direction in which the loss increases**.

---

## 2. The Key Question

```text
What do we do with this gradient?
```

* Computing $\frac{\partial \mathcal{L}}{\partial W}$ gives information, but **no actual learning happens yet**.
* Gradient alone does not update the model.

---

## 3. The Core Idea of Optimization

The basic principle of optimization in deep learning:

```text
Move parameters in the direction that reduces the loss.
```

Formally, we define an **update rule** to translate gradient information into parameter changes.

---

## 4. Gradient Descent Update Rule

The simplest update rule is **gradient descent (GD)**:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
$$

In row-vector notation for a linear layer:
$$
z = xW + b, \quad W^{(t+1)} = W^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(t)}}
$$

Where:

* $x \in \mathbb{R}^{1 \times d}$ is the input row vector
* $W \in \mathbb{R}^{d \times d_{\text{out}}}$ is the weight matrix
* $b \in \mathbb{R}^{1 \times d_{\text{out}}}$ is the bias row vector
* $\eta$ = learning rate (step size)
* $\frac{\partial \mathcal{L}}{\partial W}$ = gradient of the loss

> This is the most fundamental bridge from "having a gradient" to "actually learning".

---

## 5. Intuition Behind Gradient Descent

* Gradient is the **slope** of the loss function along the parameter dimension.
* Negative gradient points toward **steepest descent**.
* By moving a small step $\eta$ in that direction, we reduce the loss:

```text
Gradient = slope
Move downhill = reduce loss
```

---

## 6. Key Insight — Gradient ≠ Learning

Computing gradients is only **information gathering**.
**Learning happens only when we apply updates**:

$$
W^{(t+1)} = W^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(t)}}
$$

Where $t$ denotes the iteration step.

* Gradient alone tells us **where to go**.
* Update rule tells us **how far to go**.
* Step size $\eta$ controls **speed and stability** of learning.

---

## 7. Summary

1. Gradient gives **direction**, not learning.
2. Update rule translates gradient into **actual parameter changes**.
3. Gradient descent is the simplest **optimization rule**:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
$$
