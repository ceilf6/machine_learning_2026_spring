# Backpropagation — From Forward Computation to Gradients

![](./img/backpropagation_main2.gif)

---

## 1. Forward Computation

A model computes outputs as a function of inputs and parameters:

$$
\hat{y} = f(x; W, b)
$$

Where:

* $x$ is the input
* $W, b$ are model parameters
* $\hat{y}$ is the model prediction

This process is called the **forward pass**.

---

## 2. Loss Function

![](./img/backpropagation_main4.gif)

To measure the discrepancy between predictions and true values, we define a loss function:

$$
\mathcal{L} = \mathcal{L}(\hat{y}, y)
$$

Where $y$ is the true label.

The loss function quantifies the **error** of the model.

---

## 3. Learning Objective

![](./img/backpropagation_main3.gif)


The goal of learning is to **minimize the loss** with respect to the parameters:

$$
\min_{W, b} \mathcal{L}
$$

We want to find the parameter values that produce the most accurate predictions.

---

## 4. Parameter Sensitivity

Key question:

```text
How does changing a parameter affect the loss?
```

Formally, we need the **gradients**:

$$
\frac{\partial \mathcal{L}}{\partial W}, \quad \frac{\partial \mathcal{L}}{\partial b}
$$

Gradients tell us the **direction and magnitude** in which to adjust each parameter to reduce the loss.

---

## 5. Role of Gradients

* A **large gradient** means the loss is very sensitive to that parameter.
* A **small gradient** means the parameter has little effect on the loss.

Gradients are the **guiding signal** for updating parameters.

---

## 6. Challenge in Deep Models

![](./img/numofcoefficients.jpg)


For deep models, the output is a **composition of many layers**:

$$
\hat{y} = \underbrace{f^{(L)}}_{\text{output layer}}(\underbrace{f^{(L-1)}}_{\text{layer } L-1}(\dots \underbrace{f^{(1)}}_{\text{layer } 1}(\underbrace{x}_{\text{input}}) \dots ))
$$

* Each layer applies a nonlinear transformation
* Parameters are nested and interact in complex ways

Computing $\frac{\partial \mathcal{L}}{\partial W^{(l)}}$ manually for each layer is **impractical**.

