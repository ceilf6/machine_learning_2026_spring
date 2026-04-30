# Adam Optimizer — Learning to Trust the Gradient

When training neural networks, one of the most important questions is:

> **How large should each step be when updating parameters?**

This is controlled by the **learning rate**.

If the learning rate is:

* **Too small** → training becomes extremely slow
* **Too large** → optimization becomes unstable

Ideally, we would like an optimizer that can **automatically adjust step sizes during training**.

This idea eventually led to one of the most widely used optimizers in deep learning:

**Adam (Adaptive Moment Estimation)**.

But to understand Adam, we first need to understand how optimization evolved from **Gradient Descent → SGD → Momentum → Adam**.

---

# 1. Gradient Descent — Following the Slope

Standard gradient descent updates parameters using the gradient of the loss function:

$$
W^{(t+1)} = W^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(t)}}
$$

In row-vector notation for a linear layer:
$$
z = xW + b
$$

Where:

* $W$ — model parameters (weights)
* $\eta$ — learning rate
* $\frac{\partial \mathcal{L}}{\partial W}$ — gradient of the loss


![](img/0.gif)
![](img/3.gif)



---

### Problem

To compute the gradient exactly, we must evaluate the loss using **the entire dataset**.

For large datasets, this becomes extremely expensive.

---

# 2. Stochastic Gradient Descent (SGD)

Instead of using the entire dataset, **Stochastic Gradient Descent (SGD)** estimates the gradient using a **small random batch**.

$$
W^{(t+1)} = W^{(t)} - \eta g^{(t)}
$$

where

$$
g^{(t)} \approx \frac{\partial \mathcal{L}}{\partial W^{(t)}}
$$



![](./img/s1.jpg)


---

## Why SGD Works Surprisingly Well


---

### 1. Faster Updates

Full gradient descent must process the entire dataset before updating.

SGD updates **after every mini-batch**, allowing many more updates per unit time.

---

### 2. Escaping Local Minima

Because SGD is noisy, the optimization path **wiggles**.

This is often beneficial.

Imagine a person walking downhill:

* **Gradient Descent**: walking carefully and precisely
* **SGD**: walking slightly drunk, wobbling a little

Sometimes that wobble helps the person **step out of small pits**.

![](./img/s2.jpg)


---

### 3. Escaping Saddle Points

In high-dimensional optimization, **saddle points** are extremely common.

At these points:

* gradient ≈ 0
* but the location is **not actually optimal**

Full gradient descent can stall here.

SGD's randomness helps push the optimizer **away from the saddle point**.

![](./img/11.gif)

---

### But SGD Has Problems

Despite its advantages, SGD introduces several issues:

* Noisy updates
* Slow convergence
* Oscillations along steep directions

To fix this, researchers introduced **Momentum**.

---

# 3. Momentum — Accumulating Velocity

Momentum introduces the idea of **velocity** into optimization.

Instead of moving purely according to the current gradient, we accumulate a running average:

$$
v^{(t)} = \beta v^{(t-1)} + (1-\beta) g^{(t)}
$$

Then update parameters:

$$
W^{(t+1)} = W^{(t)} - \eta v^{(t)}
$$


![](./img/1.gif)

---

### Intuition: The Rolling Ball

Imagine again the ball rolling down a mountain.

Without momentum:

* every step depends only on the **current slope**

With momentum:

* the ball **accumulates speed** as it moves downhill


Once the ball gains speed, it can **roll through small bumps and obstacles**.

This helps the optimizer:

* ignore small noisy gradients
* maintain consistent direction
* move faster along long valleys

---

### Momentum as Trust in History

Momentum effectively says:

> “If the gradients have been pointing in a similar direction for a while, we should trust that direction more.”

So updates depend not only on the **current gradient**, but also on **recent history**.

---

# 4. Adaptive Learning Rates

Even with momentum, another problem remains.

Different parameters may experience **very different gradient scales**.

Some parameters may have:

* large gradients
* very small gradients

Using one global learning rate can therefore be inefficient.

This leads to the idea:

> **Each parameter should have its own effective learning rate.**

This is the core idea behind **adaptive optimizers**.

---

# 5. Adam — Adaptive Moment Estimation


![](./img/9.gif)


Adam combines **two ideas**:

1. **Momentum (first moment)**
2. **Adaptive learning rates (second moment)**

It tracks two statistics:

### First moment (average gradient)

$$
m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1) g^{(t)}
$$



### Second moment (average squared gradient)

$$
v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2) \big(g^{(t)}\big)^2
$$

---

# 6. Adam Update Rule

Adam updates parameters using:

$$W^{(t+1)} = W^{(t)} - \eta \frac{m^{(t)}}{\sqrt{v^{(t)}} + \epsilon}$$

Where:
- $m^{(t)}$ — moving average of gradients
- $v^{(t)}$ — moving average of squared gradients
- $\epsilon$ — small number for numerical stability

This formula creates parameter-specific effective learning rates.

![](./img/5.gif)

---

# 7. What the Two Moments Mean

### First Moment — Directional Memory

The first moment $m^{(t)}$ acts like **momentum**.

It answers:

> “What has been the consistent direction of movement recently?”

This helps smooth noisy gradients.


![](./img/8.gif)

---

### Second Moment — Gradient Stability

The second moment $v^{(t)}$ measures **how large and unstable gradients have been**.

Large values indicate **volatile directions**.

Adam therefore reduces step size in those directions.

![](./img/7.gif)

---

# 8. The Ship Captain Analogy

Another way to think about Adam is through a **ship captain navigating in fog**.

The captain receives direction reports from lookouts.

However, not every report should be trusted equally.

The captain keeps two mental records.

---

### Record 1 — Direction History (Momentum)

If a lookout has consistently suggested “slightly left”, the captain trusts that trend.

A sudden shout of “hard right!” will not immediately change course.

This is like **momentum smoothing gradients**.

---

### Record 2 — Reliability History (Second Moment)

Some lookouts are calm and consistent.

Others panic and constantly change directions.

If a lookout's past advice has been unstable, the captain **reduces trust in their instructions**.

This corresponds to scaling updates using $v^{(t)}$.

---

# 9. Why Adam Works Well

![](./img/13.gif)


Adam adapts both:

* **direction** (momentum)
* **step size** (adaptive learning rate)

As a result it can:

* smooth noisy gradients
* move quickly along stable directions
* reduce movement along unstable directions

This makes optimization **more stable and efficient**.

---

# 10. Bias Correction

Because moving averages start at zero, early estimates are biased.

Adam corrects this using:

$$
\hat{m}^{(t)} = \frac{m^{(t)}}{1-\beta_1^t}
$$

$$
\hat{v}^{(t)} = \frac{v^{(t)}}{1-\beta_2^t}
$$

The final update rule with bias correction becomes:

$$W^{(t+1)} = W^{(t)} - \eta \frac{\hat{m}^{(t)}}{\sqrt{\hat{v}^{(t)}} + \epsilon}$$

This improves early training stability.

---

# 11. Default Hyperparameters

Typical values are:

| Parameter     | Value |
| ------------- | ----- |
| learning rate | 0.001 |
| β₁            | 0.9   |
| β₂            | 0.999 |
| ε             | 1e-8  |

These work well across many tasks.

---

# 12. Why Adam Became So Popular

Adam became widely used because it:

* converges quickly
* requires little tuning
* handles noisy gradients well
* scales to large neural networks

For this reason many frameworks use Adam as the **default optimizer**.

---

# 13. Using Adam in PyTorch

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Training loop:

```python
for x, y in dataloader:
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
```

---

# 14. Summary

Adam combines three key ideas:

1. **SGD** — efficient stochastic updates
2. **Momentum** — accumulate directional velocity
3. **Adaptive learning rates** — parameter-specific step sizes

This allows Adam to navigate complex loss landscapes **efficiently and robustly**.
