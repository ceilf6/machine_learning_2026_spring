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
z = xW + b, \quad W^{(t+1)} = W^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(t)}}
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


* Training progresses slowly
* May require many iterations
* Inefficient and computation-heavy

### 3.2 Too Large

* Updates overshoot the optimum
* Loss may oscillate or diverge
* Training can fail entirely

### 3.3 Just Right

$$
\eta \approx \text{optimal}
$$

* Fast convergence without overshooting
* Training is stable and efficient

---

## 4. Learning as a Dynamic Process

Optimization is iterative — each step depends on the previous one:

$$
W^{(t+1)} = W^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(t)}}
$$

* **Too small** → sluggish movement in the loss landscape
* **Too large** → oscillations or divergence
* **Proper $\eta$** → smooth convergence

The learning rate essentially governs the **dynamics** of the optimization process.

---

## 5. Practical Learning Rate Values

Typical ranges depend on model and optimizer:

| Scenario                | Typical Value |
| ----------------------- | ------------- |
| Simple gradient descent | 0.01          |
| Deep neural networks    | 0.001         |
| Adaptive optimizers     | 0.001         |
| Very large models       | 0.0001        |

> Rule of thumb: Tune $\eta$ in powers of 10 to quickly find a workable range.

---

## 6. Learning Rate Schedules

Rarely do we keep $\eta$ constant. **Schedules** adapt it over training to match optimization needs.

### Early Training

* Parameters far from optimum
* Large updates accelerate progress

### Late Training

* Gradients diminish
* Overshooting is dangerous
* Smaller $\eta$ enables fine-tuning

---

## 7. Common Schedules

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

## 8. Learning Rate Warmup

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

## 9. Summary

* Learning rate $\eta$ scales gradient updates
* Too small → painfully slow learning
* Too large → instability or divergence
* Proper choice → fast, stable, efficient convergence
* Use schedules and warmup for best results
