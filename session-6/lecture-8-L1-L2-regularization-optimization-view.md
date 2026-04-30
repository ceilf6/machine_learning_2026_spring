# L1/L2 Regularization — Optimization View

---

## 1. Goal

Regularization modifies not only the objective, but also the **optimization dynamics**.

We analyze how L1 and L2 affect gradient-based learning.

---

## 2. L2 Regularization

#### Objective:

$$
\mathcal{L}_{\text{train}}(W) + \lambda \sum_{j=1}^{d} W_j^2
$$


#### Gradient

$$
\frac{\partial \mathcal{L}}{\partial W_j} = \frac{\partial \mathcal{L}_{\text{train}}}{\partial W_j} + 2\lambda W_j
$$

#### Update Rule

$$
W_j \leftarrow W_j - \eta \left( \frac{\partial \mathcal{L}_{\text{train}}}{\partial W_j} + 2\lambda W_j \right)
$$


#### Interpretation

Rewriting:

$$
W_j \leftarrow (1 - 2\eta\lambda) W_j - \eta \frac{\partial \mathcal{L}_{\text{train}}}{\partial W_j}
$$


####  Effect

* weights are continuously shrunk
* larger weights decay faster

This is why L2 is called **weight decay**.

---

## 3. L1 Regularization

#### Objective:

$$
\mathcal{L}_{\text{train}}(W) + \lambda \sum_{j=1}^{d} \|W_j\|
$$


#### Gradient

$$
\frac{\partial \mathcal{L}}{\partial W_j} = \frac{\partial \mathcal{L}_{\text{train}}}{\partial W_j} + \lambda \cdot \text{sign}(W_j)
$$


#### Update Rule

$$
W_j \leftarrow W_j - \eta \left( \frac{\partial \mathcal{L}_{\text{train}}}{\partial W_j} + \lambda \cdot \text{sign}(W_j) \right)
$$


#### Effect

* constant shrinkage toward zero
* independent of magnitude
* introduces non-smooth behavior at zero

---

## 4. Optimization Behavior Difference


#### L2

* smooth gradients
* stable updates
* gradual shrinkage

Weights rarely become exactly zero.


#### L1

* non-smooth point at $W_j = 0$
* induces thresholding behavior
* *can* drive weights exactly to zero
