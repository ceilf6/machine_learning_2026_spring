# Neural Networks — Output Layers and Softmax

![](./img/d2.jpg)

---

## 1. The Output Layer — Task-Specific Mapping

The output layer is **the final step in a neural network**, transforming learned internal representations into the form required by the task. All hidden layers before the output focus on learning **useful features**. The output layer translates those features into actionable predictions.

**Key points:**

1. The output layer defines the **dimensionality** of predictions.

   * Regression → 1 or more continuous outputs
   * Binary classification → 1 probability
   * Multiclass classification → $K$ probabilities

2. The output layer also determines **the activation function** applied to raw outputs ($z^L$).

3. The output layer is tightly coupled to the **loss function**, ensuring that learning signals correctly optimize the task.

---

## 2. Regression Output Layer

Regression tasks predict continuous values, such as house prices, temperatures, or stock prices.

### Forward Pass

For regression, the output layer is **linear**, no activation:

$$
\hat{y} = z^{(L)} = a^{(L-1)} W^{(L)} + b^{(L)}
$$

Here:

* $a^{(L-1)} \in \mathbb{R}^{1 \times n_{L-1}}$ — activations from the last hidden layer
* $W^{(L)} \in \mathbb{R}^{n_{L-1} \times d_{out}}$ — output layer weights
* $b^{(L)} \in \mathbb{R}^{1 \times d_{out}}$ — output layer bias

### Loss Function

We typically use **Mean Squared Error (MSE)**:

$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2
$$

* MSE directly measures the squared difference between predicted and true values
* Linear output ensures the network can model any real-valued target



---

## 3. Binary Classification Output Layer

Binary classification predicts whether a sample belongs to **one of two classes**.

### Forward Pass

For binary tasks, we use a **sigmoid activation**:

$$
\hat{y} = \sigma(z^{(L)}) = \frac{1}{1 + e^{-z^{(L)}}}
$$

*Output is constrained to [0,1], representing a probability.*

### Loss Function

Binary Cross-Entropy (BCE) is commonly used:

$$
\mathcal{L} = - \frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log \hat{y}^{(i)} + (1-y^{(i)}) \log (1-\hat{y}^{(i)}) \right]
$$

---

## 4. Multiclass Classification Output Layer


![](./img/3b1b-1.gif)


Multiclass tasks involve **more than two mutually exclusive classes**, e.g., digit recognition (0–9).

### Forward Pass: Softmax

Softmax converts raw scores (logits) into a valid probability distribution:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i=1,\dots,K
$$

Properties:

1. $0 \le \hat{y}_i \le 1$ for all $i$
2. $\sum_{i=1}^{K} \hat{y}_i = 1$
3. Probability for each class is **normalized**, reflecting competition among classes

### Loss Function: Categorical Cross-Entropy

$$
\mathcal{L} = - \frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log \hat{y}_{i,k}
$$

Where:

* $y_{i,k} = 1$ if sample $i$ belongs to class $k$, else 0
* $\hat{y}_{i,k}$ — predicted probability for class $k$

**Interpretation:** Cross-entropy measures the difference between the predicted distribution and the true one-hot label distribution.

---

## 6. Practical Considerations

1. **Dimensionality**: The output layer dimension must match the target variable or number of classes (for binary classifiction, we can have one neuron or two neurons).
2. **Choice of activation**:

   * Linear → regression
   * Sigmoid → binary classification
   * Softmax → multiclass classification
3. **Compatibility with loss**: Always choose a loss function that aligns with the output activation.
4. **Numerical stability**: Softmax can produce very large exponentials; in practice, we subtract the max logit before exponentiating to avoid overflow:

$$
\hat{y}_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{K} e^{z_j - \max_j z_j}}
$$
