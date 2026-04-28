# The Sigmoid Model

![](./img/lg2.jpg)


---

## 1. The missing piece

From the previous lecture, we have:

$$
z = x W + b
$$

where $x \in \mathbb{R}^{1 \times d}$ is a row vector and $W \in \mathbb{R}^{d \times 1}$ is a weight matrix.

But this is still:

$$
z \in (-\infty, +\infty)
$$

We need:

$$
\hat{y} \in [0, 1]
$$

So the question becomes:

> How do we convert $z$ into a valid probability?

---

## 2. The sigmoid function

![](./img/sigmoid2.jpg)


We introduce a function:

$$
\boxed{\sigma(z) = \frac{1}{1 + e^{-z}}}
$$

This is called the sigmoid function.

---

## 3. Why sigmoid?

The sigmoid function has exactly the properties we need.

### Bounded output

$$
\sigma(z) \in (0, 1)
$$

No matter what $z$ is, the output is always a valid probability.

---

### Smooth and differentiable

The function is continuous and smooth everywhere.

This makes it suitable for gradient-based optimization.

---

### Monotonic

If $z_1 > z_2$, then:

$$
\sigma(z_1) > \sigma(z_2)
$$

So ordering is preserved.

---

### Key values

$$
\sigma(0) = 0.5
$$

* Negative $z$ → output close to 0
* Positive $z$ → output close to 1

---

## 4. The logistic regression model

![](./img/lg-nn.gif)


Now we combine everything.

First compute:

$$
z = x W + b
$$

Then apply sigmoid:

$$
\hat{y} = \sigma(z)
$$

So the full model is:

$$
\hat{y} = \sigma(x W + b)
$$

---

## 5. Interpretation

![](./img/sigmoid.jpg)

The output now has a clear meaning:

$$
\hat{y} = \text{probability that } y = 1
$$

This gives:

* A value between 0 and 1
* A notion of confidence
* A smooth transition between classes
