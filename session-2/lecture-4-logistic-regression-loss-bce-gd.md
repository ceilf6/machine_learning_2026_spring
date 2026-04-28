# Loss Function and Gradient Descent in Logistic Regression



![](./img/logisticregressionanimatedgif5.gif)

---

## 1. Objective

We have a logistic regression model:

$$
\hat{y} = \sigma(z) = \sigma(x W + b)
$$

where $x \in \mathbb{R}^{1 \times d}$ is the feature row vector, $W \in \mathbb{R}^{d \times 1}$ is the weight matrix, $b$ is the bias, and $\sigma$ is the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The true label is binary:

$$
y \in {0, 1}
$$

Our goal is to **adjust $W$ and $b$ such that predicted probabilities $\hat{y}$ match the true labels $y$** as closely as possible across all samples.

This raises two key questions:

1. **How do we measure the error of a probability prediction?**
2. **How do we adjust the parameters to reduce that error?**

---

## 2. Why not Mean Squared Error?

A naive idea is to use the same loss as in linear regression:

$$
\mathcal{L}_\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

At first glance, this seems reasonable: $\hat{y}$ is a number, $y$ is a number, just take the squared difference.

However, MSE is **not suitable for probabilities**. Here's why:

### 2.1 Weak gradients at extremes

* The sigmoid function flattens at large positive or negative $z$:

$$
\sigma(z) \to 1 \text{ as } z \to +\infty, \quad \sigma(z) \to 0 \text{ as } z \to -\infty
$$

* When the prediction is extremely confident but wrong, the derivative of MSE w.r.t $z$ becomes very small.
* The model learns slowly or can even get stuck.

### 2.2 Poor feedback for wrong predictions

Example:

* True label $y = 1$
* Prediction $\hat{y} = 0.01$

The MSE loss is:

$$
\mathcal{L} = (\hat{y}-y)^2 = (0.01 - 1)^2 = 0.9801
$$

Its gradient w.r.t $z$ is tiny because the derivative passes through the sigmoid:

$$
\frac{d\mathcal{L}}{dz} = (\hat{y}-y) \hat{y}(1-\hat{y}) = (-0.99)(0.01*0.99) \approx -0.0098
$$

Notice how **the correction signal is tiny** despite being a huge mistake. This slows down learning drastically.

### 2.3 Ignores probability semantics (optional)

MSE treats $\hat{y}$ as a raw number. But $\hat{y}$ is a **probability**. We want a loss function that:

* Penalizes confident mistakes heavily
* Provides a strong gradient signal
* Aligns with probability theory

---

## 3. Binary Cross Entropy (BCE)

![](./img/lg.jpg)


The natural choice is **Binary Cross Entropy (BCE)**:

$$
\boxed{\mathcal{L}_\text{BCE} = - \frac{1}{n} \sum_{i=1}^{n} \big( y_i \log \hat{y}_i + (1-y_i) \log (1 - \hat{y}_i) \big)}
$$

**Intuition:**

* If $y_i = 1$, the first term $- \log \hat{y}_i$ dominates
* If $y_i = 0$, the second term $- \log (1 - \hat{y}_i)$ dominates
* Predictions close to the true label → small loss
* Confident mistakes → very large loss



---

### 3.1 Behavior of BCE

| True $$y$$ | Predicted $$\hat{y}$$ | Loss $$-y \log \hat{y} - (1-y)\log(1-\hat{y})$$ |
| -------- | ------------------- | --------------------------------------------- |
| 1        | 0.9                 | 0.105                                         |
| 1        | 0.5                 | 0.693                                         |
| 1        | 0.01                | 4.605                                         |
| 0        | 0.9                 | 2.303                                         |
| 0        | 0.5                 | 0.693                                         |
| 0        | 0.01                | 0.010                                         |

Notice how **BCE penalizes wrong confident predictions sharply**, giving strong gradient signals for learning.

---

## 4. Gradient of BCE


For a single sample:

$$
\mathcal{L}_i = - \big( y \log \hat{y} + (1-y)\log(1-\hat{y}) \big)
$$

Gradient w.r.t $z$ (pre-activation):

$$
\frac{\partial \mathcal{L}_i}{\partial z} = \hat{y} - y
$$

**Remark:** This is extremely elegant:

* Directly the difference between predicted probability and true label
* Avoids vanishing gradients
* Works naturally with the sigmoid function

---

### 4.1 Gradient w.r.t weights and bias

Since $z = x W + b$:

* Gradient w.r.t weights:

$$
\frac{\partial \mathcal{L}_i}{\partial W} = x^{\mathsf{T}} (\hat{y} - y)
$$

* Gradient w.r.t bias:

$$
\frac{\partial \mathcal{L}_i}{\partial b} = \hat{y} - y
$$

Note: With the row-vector convention ($x \in \mathbb{R}^{1 \times d}$), the gradient $\frac{\partial \mathcal{L}_i}{\partial W} \in \mathbb{R}^{d \times 1}$ has the same shape as $W$.

---

### 4.2 Intuition

* $(\hat{y} - y)$ acts as the **error signal**
* Multiplying by $x$ scales the update for each feature
* The parameter update moves in the direction that **reduces the error**

---

## 5. Gradient Descent Updates


![](./img/logisticregressionanimatedgif3.gif)


For a learning rate $\eta$, batch gradient descent:

$$
W \leftarrow W - \eta \frac{1}{n} \sum_{i=1}^{n} x_i^{\mathsf{T}} (\hat{y}_i - y_i)
$$

$$
b \leftarrow b - \eta \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

Or, in matrix form (full batch; and be used for non-full batch as well):

$$
\boxed{W \leftarrow W - \eta
\frac{1}{n}
X^{\mathsf T}
(\hat{Y}-Y)}
$$

$$
\boxed{b \leftarrow b - \eta \frac{1}{n} \sum_{i=1}^{n}
(\hat{y}_i-y_i)}
$$



---

## 6. Single-step example

Suppose:

* Feature: $x = [1, 2]$ (row vector, $x \in \mathbb{R}^{1 \times 2}$)
* Weight: $W = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$ (weight matrix, $W \in \mathbb{R}^{2 \times 1}$)
* Bias: $b = 0$
* True label: $y = 1$

1. Compute pre-activation:

$$
z = x W + b = [1, 2] \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} + 0 = 0.5*1 + (-0.5)*2 + 0 = -0.5
$$

2. Compute prediction:

$$
\hat{y} = \sigma(-0.5) \approx 0.38
$$

 3. Compute gradient w.r.t weights:

 $$
 \frac{\partial \mathcal{L}}{\partial W} = x^{\mathsf{T}} (\hat{y}-y) = \begin{bmatrix} 1 \\ 2 \end{bmatrix} (0.38 - 1) = \begin{bmatrix} -0.62 \\ -1.24 \end{bmatrix}
 $$

4. Update weights with learning rate $\eta = 0.1$:

$$
W_\text{new} = W - \eta \frac{\partial \mathcal{L}}{\partial W} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} - 0.1 \begin{bmatrix} -0.62 \\ -1.24 \end{bmatrix} = \begin{bmatrix} 0.562 \\ -0.376 \end{bmatrix}
$$

5. Update bias:

$$
b_\text{new} = b - \eta (\hat{y}-y) = 0 - 0.1*(0.38 - 1) = 0.062
$$


---

## 7. Training Loop in Practice


![](./img/logisticregressionanimatedgif.gif)


1. Forward pass: compute $z = x W + b$ and $\hat{y} = \sigma(z)$
2. Compute BCE loss
3. Backward pass: compute gradients $\partial \mathcal{L} / \partial W$ and $\partial \mathcal{L} / \partial b$
4. Update parameters with gradient descent
5. Repeat for all batches / epochs until convergence

