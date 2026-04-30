# Generalization

---

## 1. Goal

The goal of a learning algorithm is not to optimize performance on the training data. It is to achieve strong **performance on unseen data**.

This is called **generalization**.

---

## 2. Setup

Dataset:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^n
$$

Model:

$$
\hat{y}^{(i)} = f(x^{(i)})
$$

---

## 3. Training Error

Empirical loss on training data:

$$
\mathcal{L}_{\text{train}} = \frac{1}{n} \sum_{i=1}^{n} \ell(\hat{y}^{(i)}, y^{(i)})
$$

This is observable.

---

## 4. True Error

Expected loss over the data distribution:

$$
\mathcal{L}_{\text{true}} = \mathbb{E}_{(x,y)}[\ell(f(x), y)]
$$

This is what we want.

It is **unknown**.

---

## 5. Key Gap

$$
\mathcal{L}_{\text{train}} \neq \mathcal{L}_{\text{true}}
$$

A model can have:

* low $\mathcal{L}_{train}$
* high $\mathcal{L}_{true}$

This is **overfitting**.

---

## 6. Intuition

The model can:

* memorize $\mathcal{D}$
* fit noise instead of signal

Result:

* good on seen data
* bad on unseen data
