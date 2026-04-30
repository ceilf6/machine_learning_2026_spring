# Dropout

---

## 1. Motivation: Why Regularization is Needed

Neural networks often contain a large number of parameters. This makes them highly expressive but also prone to overfitting.

A model with high capacity can:

* memorize training samples
* fit noise in the data
* fail to generalize to unseen inputs

Dropout is a regularization technique designed to reduce this effect by preventing co-adaptation of neurons.

---

## 2. Core Idea of Dropout

During training, we randomly deactivate a subset of neurons.

For each neuron:

$$
m_i \sim \text{Bernoulli}(1 - p)
$$

where:

* $p$ is the dropout probability
* $m_i = 0$ means the neuron is dropped
* $m_i = 1$ means the neuron is kept

The forward pass becomes:

$$
\tilde{h} = h \odot m
$$

where:

* $h$ is the original activation
* $m$ is the binary mask
* $\odot$ is element-wise multiplication

---

## 3. Inverted Dropout

In practice, we use inverted dropout to maintain consistent activation scale.

We define the mask as:

$$
m_i \sim \frac{\text{Bernoulli}(1 - p)}{1 - p}
$$

Then:

$$
\tilde{h} = h \odot m
$$

### Key property

The expectation remains unchanged:

$$
\mathbb{E}[\tilde{h}] = h
$$

This ensures that:

* training and inference operate on the same scale
* no rescaling is needed during inference

---

## 4. Intuition: Implicit Ensemble Learning

Dropout can be interpreted as training a large ensemble of sub-networks.

Each forward pass samples a different sub-network:

* different subsets of neurons are active
* parameters are shared across all sub-networks

This leads to:

* model averaging effect
* improved robustness
* reduced overfitting

---

## 5. Effect on Representation Learning

Without dropout, neurons can develop strong dependencies.

With dropout:

* neurons must work independently
* redundant representations are learned
* features become more robust

This reduces co-adaptation and improves generalization.

---

## 6. Dropout in Neural Networks

A typical layer with dropout:

$$
h = \sigma(xW + b)
$$

During training:

$$
\tilde{h} = h \odot m
$$

During inference:

$$
\tilde{h} = h
$$

No dropout is applied at inference time.

---

## 7. PyTorch Implementation

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10)
)
```

Here:

* $p = 0.5$ means 50% dropout rate
* dropout is active only during training mode

---

## 8. Geometric Interpretation

Dropout injects noise into hidden representations:

* training becomes stochastic
* decision boundaries are smoothed
* sharp fitting is discouraged

From a function perspective:

* the model learns a family of functions
* robustness is enforced across perturbations of hidden units
