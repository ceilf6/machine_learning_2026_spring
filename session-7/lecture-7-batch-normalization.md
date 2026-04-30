# Batch Normalization

---

## 1. Motivation: Internal Covariate Shift

During training of deep neural networks, the distribution of activations in intermediate layers keeps changing as parameters update.

Consider a layer:

$$
z = xW + b
$$

As training progresses:

* input distribution to each layer shifts
* later layers must continuously adapt
* optimization becomes unstable

This phenomenon motivates Batch Normalization.

---

## 2. Core Idea

Batch Normalization standardizes intermediate activations within a mini-batch.

Given a batch:

$$
z_1, z_2, \dots, z_m
$$

Compute batch statistics:

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} z_i
$$

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (z_i - \mu_B)^2
$$

Normalize:

$$
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

---

## 3. Learnable Transformation

After normalization, we restore representation flexibility using:

* scale parameter $\gamma$
* shift parameter $\beta$

Final output:

$$
y_i = \gamma \hat{z}_i + \beta
$$

This ensures the network can recover any necessary distribution if normalization is not optimal.

---

## 4. Intuition

Batch Normalization does two things:

* stabilizes activation distribution
* reduces sensitivity to parameter initialization

Effectively:

* each layer receives inputs with controlled mean and variance
* optimization becomes easier and more predictable

---

## 5. Placement in Neural Networks

Standard structure:

$$
\text{Linear} \rightarrow \text{BatchNorm} \rightarrow \text{Activation}
$$

Example:

$$
xW + b \rightarrow \text{BN} \rightarrow \text{ReLU}
$$

This ensures normalization happens before non-linearity.

---

## 6. Training Behavior

During training:

* statistics are computed per mini-batch
* each batch introduces slight randomness

Running estimates are maintained:

$$
\mu_{running} \leftarrow (1 - \alpha)\mu_{running} + \alpha \mu_B
$$

$$
\sigma^2_{running} \leftarrow (1 - \alpha)\sigma^2_{running} + \alpha \sigma_B^2
$$

These approximate global dataset statistics.

---

## 7. Inference Behavior

During inference:

* batch statistics are not reliable
* use running estimates instead

Normalization becomes:

$$
\hat{z} = \frac{z - \mu_{running}}{\sqrt{\sigma^2_{running} + \epsilon}}
$$

This ensures deterministic behavior.

---

## 8. Effects on Optimization

Batch Normalization improves training in several ways:

### 8.1 Faster Convergence

* allows higher learning rates
* reduces sensitivity to initialization

---

### 8.2 Gradient Stability

* reduces exploding gradients
* mitigates vanishing gradients

---

### 8.3 Implicit Regularization

Because batch statistics vary:

* noise is injected during training
* model becomes more robust
* overfitting is reduced

---

## 9. Relation to Standardization

Standardization applies to input features:

$$
x \rightarrow \frac{x - \mu}{\sigma}
$$

Batch Normalization applies the same idea internally:

* inside hidden layers
* during training dynamics

Thus it can be viewed as:

> standardization applied repeatedly inside a deep network

---

## 10. Summary

Batch Normalization:

* stabilizes intermediate activations
* improves optimization speed
* reduces sensitivity to initialization
* provides regularization effect

It is a core technique for training deep neural networks efficiently.


