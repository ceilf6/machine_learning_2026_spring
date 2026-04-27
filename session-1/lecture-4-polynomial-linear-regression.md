# Polynomial Linear Regression

![](./img/p.jpg)

---

# 1. Motivation

So far, we assumed a **linear relationship**:

$$
\hat{y} = w x + b
$$

But many real-world relationships are **not linear**.

Examples:

* growth curves
* physical systems
* nonlinear trends in data

---

# 2. Key Idea

We keep the model **linear in parameters**, but transform the input.

We create new features:

$$
x, x^2, x^3, \dots, x^k
$$

Then apply linear regression on these features.

---

# 3. Model Formulation

For degree $k$, the model becomes:

$$
\hat{y} = w_1 x + w_2 x^2 + \cdots + w_k x^k + b
$$

This is called **polynomial regression**.

Even though the curve is nonlinear in $x$, it is still **linear in $w$**.

---

# 4. Feature Transformation

Original input:

$$
x
$$

Transformed input:

$$
\phi(x) = [x, x^2, x^3, \dots, x^k]
$$

Now we apply multiple linear regression:

$$
\hat{y} = \phi(x) W + b
$$

So polynomial regression is just **linear regression on transformed features**.

---

# 5. Example

![](./img/p2.jpg)


Degree 2 (quadratic):

$$
\hat{y} = w_1 x + w_2 x^2 + b
$$

Degree 3 (cubic):

$$
\hat{y} = w_1 x + w_2 x^2 + w_3 x^3 + b
$$

Higher degree → more flexible model.

---

# 6. Gradient Descent

We still use gradient descent.

The loss is the same:

$$
\mathcal{L}(W, b)  = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2
$$

Gradients are computed with respect to each parameter:

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)}) x^{(i)}_j
$$

Training is the same as multiple linear regression.

---

# 7. Underfitting vs Overfitting (later sessions)

Model complexity depends on the degree $k$.

Low degree:

* too simple
* cannot fit the data well
* underfitting

High degree:

* very flexible
* fits noise
* overfitting 
