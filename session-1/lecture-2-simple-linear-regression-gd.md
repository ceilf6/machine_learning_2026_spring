# Simple Linear Regression — Gradient Descent

---

# 1. Recap

In **linear regression**, we predict:

$$
\hat{y} = w x + b
$$

We want to find $w$ and $b$ that **minimize the error** between predictions and true values.

The error is measured using **Mean Squared Error (MSE)**:

$$
\boxed{\mathcal{L}(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2}
$$

![](./img/mse.jpg)

---

# 2. Why Gradient Descent

![](./img/lr_0.03.gif)


Gradient descent is a method to **iteratively update parameters** to reduce loss.

The idea:

* Compute **gradient of loss** with respect to parameters
* Take a **small step in the opposite direction** of the gradient

For $w$ and $b$:

$$
w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}
$$

$$
b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
$$

where $\eta$ is the **learning rate**.

---

# 3. Gradients for Linear Regression

![](./img/18.gif)

For MSE, the derivatives are:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)}) x^{(i)}}
$$

$$
\boxed{\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})}
$$

These tell us **how to change $w$ and $b$** to reduce error.

---

# 4. Gradient Descent Algorithm

![](./img/3.gif)

1. Initialize $w$ and $b$ (e.g., $0$)

2. Repeat for a number of iterations:

   * Compute predictions: $\hat{y}^{(i)} = w x^{(i)} + b$
   * Compute gradients: $\frac{\partial \mathcal{L}}{\partial w}$, $\frac{\partial \mathcal{L}}{\partial b}$
   * Update parameters:

   $$
   w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}
   $$

   $$
   b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
   $$

3. Stop when loss **converges** or reaches **max iterations**

---

# 5. Python Example

```python
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 9])

# Initialize parameters
w, b = 0.0, 0.0
eta = 0.01  # learning rate
iterations = 1000
n = len(X)

for _ in range(iterations):
    y_hat = w * X + b
    dw = (2/n) * np.sum((y_hat - y) * X)
    db = (2/n) * np.sum(y_hat - y)
    
    w -= eta * dw
    b -= eta * db

print("w:", w)
print("b:", b)
```

