#  Adam (Adaptive Moment Estimation)

---

## 1. Motivation

From the previous lecture, **momentum** accelerates learning along consistent directions and reduces oscillations.

However, there is another issue:

* Different parameters can have **very different gradient scales**
* Using a single global learning rate may be inefficient

We need an optimizer that combines:

1. **Directional memory** (momentum)
2. **Parameter-specific adaptive step sizes**

This leads to **Adam (Adaptive Moment Estimation)**.

---

## 2. Mini-Batch Gradient Descent Recap

For reference, mini-batch gradient descent updates parameters as:

$$
W^{(t+1)} = W^{(t)} - \eta g^{(t)}, \quad g^{(t)} = \frac{1}{B} \sum_{i \in \mathcal{B}} \frac{\partial \mathcal{L}_i}{\partial W^{(t)}}
$$

In row-vector notation for a linear layer:
$$
z = xW + b
$$

* Momentum adds a **velocity term** $v^{(t)}$
* Adam goes further by also adapting the **learning rate per parameter**

---

## 3. Adam — First and Second Moments

Adam tracks two moving averages:

### 3.1 First Moment — Mean of Gradients

$$
m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1) g^{(t)}
$$

* Similar to momentum
* Captures **directional trend** of gradients
* $\beta_1$ typically 0.9

---

### 3.2 Second Moment — Mean of Squared Gradients

$$
v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2) \big(g^{(t)}\big)^2
$$

* Measures **gradient magnitude and variability**
* Large $v$ → volatile gradients → reduce step size
* $\beta_2$ typically 0.999

---

## 4. Bias Correction

Moving averages start at 0, causing **initial bias**.

Corrected moments:

$$
\hat{m}^{(t)} = \frac{m^{(t)}}{1-\beta_1^t}
$$

$$
\hat{v}^{(t)} = \frac{v^{(t)}}{1-\beta_2^t}
$$

* Ensures **early updates are not underestimated**
* Important for stable training at the beginning

---

## 5. Adam Update Rule

Parameter update:

$$
W^{(t+1)} = W^{(t)} - \eta \frac{\hat{m}^{(t)}}{\sqrt{\hat{v}^{(t)}} + \epsilon}
$$

Where:

* $W^{(t)}$ — parameters
* $\eta$ — base learning rate (typical 0.001)
* $\hat{m}^{(t)}$ — bias-corrected first moment
* $\hat{v}^{(t)}$ — bias-corrected second moment
* $\epsilon$ — small number for numerical stability (e.g., $10^{-8}$)

**Key idea:** Step size is **adapted per parameter** based on gradient history.

---

## 6. Geometric Intuition

1. **First moment $m$** → smooths noisy gradients (like momentum)
2. **Second moment $v$** → scales updates according to gradient volatility
3. **Combined effect** → move quickly along flat, stable directions and cautiously along steep, noisy directions

Analogy: A ship navigating in fog:

* Trust **consistent directions** from history → first moment
* Reduce trust in **unreliable directions** → second moment

---

## 7. Practical Defaults

| Hyperparameter       | Typical Value |
| -------------------- | ------------- |
| Learning rate $\eta$ | 0.001         |
| $\beta_1$            | 0.9           |
| $\beta_2$            | 0.999         |
| $\epsilon$           | 1e-8          |

* Works well for most deep learning tasks
* Minimal tuning required

---

## 8. Why Adam Works Well

* Combines **SGD efficiency** with **momentum smoothing**
* Adapts **step size per parameter**
* Handles **noisy gradients**, **saddle points**, and **high-curvature landscapes**
* Converges quickly and robustly

**Takeaway:** Adam is a **safe, powerful default optimizer** for deep networks.

---

## 9. PyTorch Example

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

for x, y in dataloader:
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
```

---

## 10. Summary

Adam is the natural culmination of **mini-batch SGD + momentum + adaptive learning rates**:

1. Tracks **first moment** → smooths gradients (direction)
2. Tracks **second moment** → scales steps (magnitude)
3. Bias correction → stabilizes early training

> Adam allows neural networks to **learn efficiently and robustly** across complex, high-dimensional loss landscapes.
