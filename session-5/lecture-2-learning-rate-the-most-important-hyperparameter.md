# Learning Rate: The Most Important Hyperparameter


First, let's open those URLs for visualization/play:
- https://uclaacm.github.io/gradient-descent-visualiser/#playground

---

## 1. Gradient Descent Review

![](./img/gd.jpg)


Recall the basic gradient descent update:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
$$

In row-vector notation for a linear layer:
$$
z = xW + b, \quad W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
$$

* $W$ — model parameters (weight matrix)
* $\eta$ — learning rate
* $\frac{\partial \mathcal{L}}{\partial W}$ — gradient

**Key insight:** The gradient indicates **direction**, while the learning rate controls **step size**. Choosing $\eta$ carefully is essential for efficient training.

---

## 2. What Does the Learning Rate Do?

The learning rate $\eta$ is a **scalar hyperparameter** that scales updates in parameter space. It directly affects:

* **Speed of convergence** — how fast the model learns
* **Stability** — whether training oscillates or diverges

Intuitively:

* Small $\eta$ → tiny steps, slow progress
* Large $\eta$ → large steps, risk of overshooting

---

## 3. Learning Rate Scenarios

### 3.1 Too Small

![](./lr_0.001.gif)

* Training progresses slowly
* May require many iterations
* Inefficient and computation-heavy

### 3.2 Too Large

![](./lr_1.gif)

* Updates overshoot the optimum
* Loss may oscillate or diverge
* Training can fail entirely

### 3.3 Just Right

![](./lr_0.03.gif)

$$
\eta \approx \text{optimal}
$$

* Fast convergence without overshooting
* Training is stable and efficient

---

## 4. Practical Learning Rate Values

Typical ranges depend on model and optimizer:

| Scenario                | Typical Value |
| ----------------------- | ------------- |
| Simple gradient descent | 0.01          |
| Deep neural networks    | 0.001         |
| Adaptive optimizers     | 0.001         |
| Very large models       | 0.0001        |


---

## 5. Learning Rate Schedules

Rarely do we keep $\eta$ constant. **Schedules** adapt it over training to match optimization needs.

### Early Training

* Parameters far from optimum
* Large updates accelerate progress

### Late Training

* Gradients diminish
* Overshooting is dangerous
* Smaller $\eta$ enables fine-tuning

---

## 6. Common Schedules

### Step Decay

Reduce $\eta$ at predefined epochs:

```
epoch 0–30   lr = 0.01
epoch 30–60  lr = 0.001
epoch 60–90  lr = 0.0001
```

Simple yet effective.

### Exponential Decay

Smooth decay over time:

$$
\eta_t = \eta_0 e^{-k t}
$$

Gradually decreases step size.

### Cosine Decay

Smoothly varies $\eta$ following a cosine curve:

* Avoids sudden jumps
* Empirically effective in modern deep learning

---

## 7. Learning Rate Warmup

Large models often use a **warmup phase**:

* Start with a very small learning rate
* Gradually increase to the target value
* Then follow a decay schedule

Example:

```
step 0–2000 → increase lr
step 2000+  → decay schedule
```

Why it works:

* Early gradients can be unstable
* Large steps risk divergence
* Gradual ramp-up stabilizes training

Warmup is now standard in large-scale models.

---

## 8. Summary

* Learning rate $\eta$ scales gradient updates
* Too small → painfully slow learning
* Too large → instability or divergence
* Proper choice → fast, stable, efficient convergence
* Use schedules and warmup for best results
