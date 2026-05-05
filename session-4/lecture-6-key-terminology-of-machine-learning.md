# Core Terminology of Machine Learning

---

## 1. Inputs and Data

A machine learning model starts with data:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^n
$$

### Key Terms

* **Input / Feature / Variable**:
  $x \in \mathbb{R}^d$ — the information we use to make predictions
* **Feature Vector**:
  A collection of features, e.g. $x = (x_1, ..., x_d)$
* **Sample / Instance / Observation**:
  One data point $(x_i, y_i)$

---

## 2. Outputs and Predictions

Each input has a corresponding target:

$$
y \quad \text{and} \quad \hat{y} = f(x)
$$

### Key Terms

* **Target / Label / Ground Truth ($y$)**
  The true value we want to learn

* **Prediction / Output / Estimate ($\hat{y}$)**
  The model’s output

---

## 3. Model and Parameters

The model defines how inputs map to outputs:

$$
\hat{y} = f(x; \theta)
$$

### Key Terms

* **Model / Function / Hypothesis**
  The mapping from $x$ to $\hat{y}$

* **Parameters ($\theta$)**
  The learnable components

* **Weights**
  Parameters in neural networks

* **Coefficients**
  Parameters in linear regression

* **Bias / Intercept ($b$)**
  Constant offset in the model

---

## 4. Neural Network Terminology


* **Neural Network (NN) / Artificial Neural Network (ANN)**
  A model composed of multiple layers of parameterized functions

* **Feedforward Network (FFN)**
  A network where information flows strictly from input to output (no cycles)

* **Multilayer Perceptron (MLP)**
  A standard fully-connected feedforward neural network

* **Layer**
  A transformation of the form:
  $$
  h = \sigma(xW + b)
  $$

* **Hidden Layer**
  Intermediate layers between input and output

* **Activation Function**
  Nonlinear function such as $\sigma(\cdot)$, ReLU, etc.

* **Forward Pass**
  The computation of $\hat{y}$ from $x$ through the network

---

## 5. Loss and Optimization

Learning is defined as minimizing error:

$$
\min_\theta \; \mathcal{L}(y, \hat{y})
$$

### Key Terms

* **Error**:
  Difference between prediction and truth ($\hat{y} - y$)

* **Loss Function/Cost / Objective**:
  Measures how bad the prediction is (aggregated over the dataset)

* **Optimization**:
  The process of finding best parameters

* **Gradient Descent**:
  A method to update parameters:
  $$
  W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
  $$

---

### Final Insight

> Machine learning is the process of learning parameters $\theta$
> so that a function $f(x; \theta)$ produces predictions $\hat{y}$
> that match targets $y$ by minimizing a loss,
> while generalizing to new data.
