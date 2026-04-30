# Standardization

---

## 1. Motivation: Why Feature Scaling Matters

Many machine learning models are sensitive to the scale of input features.

Consider a feature vector:

$$
x \in \mathbb{R}^{1 \times d}
$$

If different features have different magnitudes:

* optimization becomes unstable
* gradient descent becomes inefficient
* some features dominate others

Example:

* house price prediction

  * area: 100–300 m²
  * number of rooms: 1–5

Without scaling, “area” dominates the learning process.

---

## 2. Core Idea

Standardization transforms each feature to have:

* zero mean
* unit variance

For each feature $x_j$:

$$
\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}
$$

$$
\sigma_j^2 = \frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2
$$

Then transform:

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

---

## 3. Interpretation

After standardization:

* mean becomes approximately $0$
* variance becomes approximately $1$

So each feature is on a comparable scale.

This ensures:

* no feature dominates due to magnitude
* optimization landscape becomes better conditioned

---

## 4. Why It Helps Optimization

Consider a linear model:

$$
y = xW + b
$$

If features are unscaled:

* gradients along large-scale features dominate
* optimization path becomes skewed

After standardization:

* gradient directions are more balanced
* convergence is faster and more stable

This is especially important for:

* gradient descent
* logistic regression
* neural networks (input layer sensitivity)

---

## 5. Standardization vs Normalization

### Standardization

$$
x \rightarrow \frac{x - \mu}{\sigma}
$$

* centers data at 0
* scales variance to 1
* works well for most ML models

---

### Normalization (Min-Max)

$$
x \rightarrow \frac{x - x_{min}}{x_{max} - x_{min}}
$$

* maps data to fixed range
* often used in image preprocessing

---

## 6. Important Practical Rule

Statistics must be computed only on training data:

* compute $\mu$, $\sigma$ on training set
* apply same transformation to validation/test set

Otherwise:

* data leakage occurs
* evaluation becomes invalid

---

## 7. Connection to Optimization Geometry

Without standardization:

* loss contours are elongated
* gradient descent zig-zags

With standardization:

* loss contours become more spherical
* optimization is more direct

This improves:

* convergence speed
* numerical stability

---

## 8. Summary

Standardization is a fundamental preprocessing step that:

* ensures comparable feature scales
* improves optimization efficiency
* stabilizes gradient-based learning

It is often a prerequisite for many machine learning algorithms, especially those sensitive to feature magnitude.


