# Simple Linear Regression from Scratch

## Introduction

In this tutorial, we will explore the mathematical foundations of simple linear regression, implement it from scratch using Python and NumPy.

## Mathematical Foundations

### The Linear Model

At the core of linear regression is the linear model using the **row-vector convention**:

$$
y = \mathbf{x} W + b
$$

- $y$: Predicted value (scalar)
- $\mathbf{x} \in \mathbb{R}^{1 \times d}$: Input feature row vector
- $W \in \mathbb{R}^{d \times 1}$: Weight matrix
- $b$: Bias term (scalar)

The objective is to find the optimal weights $W$ and bias $b$ that minimize the difference between the predicted values and the actual target values.

<details>
<summary>❓ Can a linear model capture non-linear relationships in data? Why or why not?</summary>

A standard linear model cannot directly capture non-linear relationships in data, but there are important nuances to understand:

1. **Fundamental Limitation**: By definition, a linear model can only produce outputs that are linear combinations of its inputs, which means it can only fit straight lines or hyperplanes.

2. **Linear Decision Boundaries**: Linear models can only create linear decision boundaries, making them unable to capture curved or complex patterns in data.

3. **Feature Engineering Workaround**: However, we can indirectly model non-linear relationships by:
   - Adding polynomial features (x², x³, etc.)
   - Creating interaction terms (x₁×x₂)
   - Applying non-linear transformations (log(x), sin(x), etc.)

For truly complex patterns, non-linear models (decision trees, neural networks) are more appropriate.
</details><br>

### Cost Function

To quantify the difference between predicted and actual values, we use the Mean Squared Error (MSE) as the cost function:

$$
\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^{N} (y^{(i)} - (\mathbf{x}^{(i)} W + b))^2
$$

Where:
- $N$: Number of samples
- $y^{(i)}$: Actual target value for the $i$-th sample
- $\mathbf{x}^{(i)} \in \mathbb{R}^{1 \times d}$: Input feature row vector for the $i$-th sample

### Gradient Descent

To minimize the cost function $\mathcal{L}(W, b)$, we employ the Gradient Descent optimization algorithm. Gradient Descent iteratively updates the weights and bias in the direction that reduces the cost.

<details>
<summary>❓ What's the intuition behind gradient descent and why does it work?</summary>

Gradient descent works like walking downhill to reach the valley:

1. The gradient points in the steepest uphill direction
2. Taking steps in the negative gradient direction moves toward lower cost
3. Steps get smaller near the minimum as the gradient approaches zero
4. For convex functions (like MSE), this guarantees finding the global minimum

</details><br>

The gradients of the cost function with respect to the weights and bias are:

$$
\frac{\partial \mathcal{L}}{\partial W} = -\frac{2}{N} \sum_{i=1}^{N} \mathbf{x}^{(i)T} (y^{(i)} - \mathbf{x}^{(i)} W - b)
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y^{(i)} - \mathbf{x}^{(i)} W - b)
$$

## How are these gradient equations derived from the cost function?

Deriving the gradient equations from the MSE cost function involves applying calculus to find the partial derivatives. Here's the step-by-step derivation:

Starting with the MSE loss function:

$$\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^{N} (y^{(i)} - (\mathbf{x}^{(i)} W + b))^2$$

Let's use the shorthand $\hat{y}^{(i)} = \mathbf{x}^{(i)} W + b$ for the prediction.

**1. Derivation of $\frac{\partial \mathcal{L}}{\partial W}$:**

Applying the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial W}$$

First part:
$$\frac{\partial \mathcal{L}}{\partial \hat{y}^{(i)}} = \frac{\partial}{\partial \hat{y}^{(i)}} \frac{1}{N} \sum_{i=1}^{N} (y^{(i)} - \hat{y}^{(i)})^2$$

$$= \frac{1}{N} \sum_{i=1}^{N} \frac{\partial}{\partial \hat{y}^{(i)}} (y^{(i)} - \hat{y}^{(i)})^2$$

$$= \frac{1}{N} \sum_{i=1}^{N} 2(y^{(i)} - \hat{y}^{(i)})(-1)$$

$$= -\frac{2}{N} \sum_{i=1}^{N} (y^{(i)} - \hat{y}^{(i)})$$

Second part:
$$\frac{\partial \hat{y}^{(i)}}{\partial W} = \frac{\partial}{\partial W} (\mathbf{x}^{(i)} W + b) = \mathbf{x}^{(i)T}$$

Combining:
$$\frac{\partial \mathcal{L}}{\partial W} = -\frac{2}{N} \sum_{i=1}^{N} (y^{(i)} - \hat{y}^{(i)}) \cdot \mathbf{x}^{(i)T}$$

$$= -\frac{2}{N} \sum_{i=1}^{N} \mathbf{x}^{(i)T} (y^{(i)} - \mathbf{x}^{(i)} W - b)$$

**2. Derivation of $\frac{\partial \mathcal{L}}{\partial b}$:**

Similarly:
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial b}$$

We already calculated the first part. For the second part:
$$\frac{\partial \hat{y}^{(i)}}{\partial b} = \frac{\partial}{\partial b} (\mathbf{x}^{(i)} W + b) = 1$$

Combining:
$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y^{(i)} - \hat{y}^{(i)}) \cdot 1$$

$$= -\frac{2}{N} \sum_{i=1}^{N} (y^{(i)} - \mathbf{x}^{(i)} W - b)$$


Using these gradients, the update rules for weights and bias are:

$$
W \leftarrow W - \eta \times \frac{\partial \mathcal{L}}{\partial W}
$$

$$
b \leftarrow b - \eta \times \frac{\partial \mathcal{L}}{\partial b}
$$

Where $\eta$ is the learning rate, a hyperparameter that controls the step size in each update.


## Implementing Linear Regression in Python

Let's implement the linear regression model from scratch using Python and NumPy. We will start with a basic implementation and then enhance it for better performance and usability.


### Initial Implementation

Below is the initial implementation of the `MyOwnLinearRegression` class, which includes methods for fitting the model to data and making predictions.

```python
%matplotlib inline
import numpy as np

class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # Predict the target values
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Example usage
regressor = MyOwnLinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Visualize the results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

### Code Explanation 

#### Initialization 

The `__init__` method initializes the learning rate, number of iterations, and placeholders for weights and bias.

#### Fit Method (`fit`)

The `fit` method trains the model using gradient descent:

1. **Parameter Initialization:**
   - Weights are initialized to zeros.
   - Bias is initialized to zero.

2. **Gradient Descent Loop:**
   - For a specified number of iterations:
     - **Prediction:** Calculate the predicted values using the current weights and bias.
     - **Gradient Calculation:** Compute the gradients of the cost function with respect to weights and bias.
     - **Parameter Update:** Update the weights and bias by moving them in the opposite direction of the gradients.


#### Predict Method (`predict`)

The `predict` method generates predictions using the trained weights and bias:

$$
\hat{y} = \mathbf{x} W + b
$$

