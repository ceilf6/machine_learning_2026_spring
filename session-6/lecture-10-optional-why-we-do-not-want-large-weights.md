# Why We Don’t Want Large Weights (Optional)

---

## 1. A First Question: Why Care About Weight Magnitude?

In almost all parametric models, we learn a function of the form

$$
f(x) = x W + b
$$

or some nonlinear extension of it.

where $x \in \mathbb{R}^{1 \times d}$ is the input row vector and $W \in \mathbb{R}^{d \times 1}$ is the weight matrix.

At first glance, it may seem that as long as the model fits the data well, the actual magnitude of the weights $W$ should not matter.

However, in practice — across **linear regression**, **logistic regression**, and **neural networks** — large weights are often a signal of deeper problems.

---

## 2. Linear Regression: Sensitivity and Instability

Consider standard linear regression:

$$
y = x W + b
$$

We learn $W$ by minimizing squared error:

$$
\min_{W} \sum_{i=1}^n (y^{(i)} - x^{(i)} W)^2
$$

### 2.1 Large Weights = High Sensitivity

If some components of $W$ are very large, then small changes in input $x$ lead to large changes in prediction.

Let’s look at a perturbation:

$$
x \rightarrow x + \delta
$$

Then the prediction changes by:

$$
f(x + \delta) - f(x) = \delta W
$$

If $\|W\|$ is large, even a tiny $\|\delta\|$ can produce a large output shift.

Interpretation:

* The model becomes **highly sensitive to noise**
* Predictions become **unstable**

---

### 2.2 Ill-Conditioning and Numerical Issues

When solving linear regression via normal equations:

$$
W = (X^T X)^{-1} X^T y
$$

If $X X^T$ is close to singular:

* The inverse explodes
* Weights become very large

Large weights often indicate:

* Multicollinearity
* Poor feature scaling
* Ill-conditioned systems

---

## 3. Logistic Regression: Confidence Explosion

Logistic regression models probability:

$$
p(y=1|x) = \sigma(x W + b)
$$

where

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

### 3.1 What Happens When Weights Are Large?

If $\|W\|$ becomes large, then for most inputs:

$$
z = x W + b \rightarrow \pm \infty
$$

Thus:

$$
\sigma(z) \rightarrow 0 \text{ or } 1
$$

The model becomes **extremely confident**.

---

### 3.2 Why Is This Bad?

#### (1) Overconfidence

The model outputs probabilities like:

* 0.999999
* 0.000001

Even for slightly uncertain inputs.

This harms:

* Calibration
* Generalization


---

## 4. Neural Networks: Amplification and Chaos


### 4.1 Layer-by-Layer Amplification

If weights are large, each layer amplifies signals:

$$
x \rightarrow x W_1 \rightarrow x W_1 W_2 \rightarrow \cdots
$$

Effect:

* Exploding activations
* Numerical instability

---

### 4.2 Gradient Explosion

Backpropagation involves products of weight matrices:

$$
\frac{\partial \mathcal{L}}{\partial x} \sim \cdots W_2 W_1
$$

If weights are large:

* Gradients grow exponentially
* Training becomes unstable

---

### 4.3 Loss Surface Becomes Sharp

Large weights correspond to functions that change very rapidly.

This leads to:

* Sharp minima
* Poor generalization

There is a strong empirical observation:

> Flat minima generalize better than sharp minima.

Large weights tend to produce **sharp curvature** in the loss landscape.

---

### 4.4 Overfitting in High Capacity Models

Neural networks already have high expressive power.

Large weights allow:

* Extreme bending of decision boundaries
* Memorization of training data

Result:

* Near-zero training error
* Poor test performance
