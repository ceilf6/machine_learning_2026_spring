# Neural Networks — From Logistic Regression to Deep Models


---

## 1. Revisiting Logistic Regression


![](./img/sigmoid2.jpg)

We start from a familiar model:

$$
\hat{y} = \sigma(x W + b)
$$

where:

* $x \in \mathbb{R}^{1 \times d}$ is the input row vector
* $W \in \mathbb{R}^{d \times 1}$ is the weight matrix
* $b \in \mathbb{R}^{1 \times 1}$ is the bias
* $\sigma(\cdot)$ is the sigmoid function

---

### Interpretation

```text
Linear transformation → nonlinearity → prediction
```

Step-by-step:

1. Compute a linear score:
   $$
   z = x W + b
   $$

2. Apply a nonlinear function:
   $$
   \hat{y} = \sigma(z)
   $$

---

## 2. A Key Reinterpretation

![](./img/lg-nn.gif)


```text
A logistic regression model is a single neuron.
```

This is the most important conceptual shift.


---

## 3. From One Neuron to Many Neurons

![](./img/ff.gif)

What if we don’t compute just one output?

```text
Instead of one neuron, we use multiple neurons in parallel.
```

---

### Vectorized form

$$
z = x W + b
$$

$$
a = \sigma(z)
$$

where:

* $x \in \mathbb{R}^{1 \times d}$
* $W \in \mathbb{R}^{d \times m}$
* $b \in \mathbb{R}^{1 \times m}$
* $z \in \mathbb{R}^{1 \times m}$
* $a \in \mathbb{R}^{1 \times m}$

---

### Interpretation

```text
Each column of W = one neuron
```

So we now have:

```text
Multiple neurons → multiple features
```

This is called a **layer**.

---

## 4. From a Layer to Multiple Layers

![](./img/d2.jpg)


Now we take the next step.

```text
What if the output of one layer becomes the input to another?
```

---

### Composition

First layer:

$$
a^{(1)} = \sigma(x W^{(1)} + b^{(1)})
$$

Second layer:

$$
a^{(2)} = \sigma(a^{(1)} W^{(2)} + b^{(2)})
$$

---

### Key insight

```text
We are stacking transformations.
```

Each layer transforms the representation learned by the previous layer.

---

## 5. The Emergence of Neural Networks

By stacking layers, we get:

$$
a^{(L)} = f(x)
$$

where $f$ is a composition of functions:

$$
f(x) = f^{(L)}(f^{(L-1)}(\cdots f^{(1)}(x)))
$$

---

### Core Definition

```text
A neural network is a composition of functions.
```

Each function is:

```text
linear transformation + nonlinearity
```
