# Neural Networks — Architecture and Notation

![](./img/d2.jpg)


---

## 1. Neural Network Structure

A neural network is organized into **layers**:

$$
\text{Input layer} \rightarrow \text{Hidden layers} \rightarrow \text{Output layer}
$$


**Input Layer**

$$
a^{(0)} = x
$$

* Receives raw input
* No computation beyond passing values

**Hidden Layers**

* Apply linear transformations followed by nonlinear activations
* Can be multiple layers stacked

**Output Layer**

* Produces final prediction
* Depends on the task (regression, binary classification, multiclass classification)

---

## 2. Standard Notation



**Layer index**

$$
l = 1, 2, \dots, L
$$

where $L$ is the total number of layers (excluding input layer).

**Linear Transformation**

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
$$

**Activation Function**

$$
a^{(l)} = g^{(l)}(z^{(l)})
$$

**Full Layer Computation**

$$
a^{(l)} = g^{(l)}(a^{(l-1)} W^{(l)} + b^{(l)})
$$

---

## 3. Dimensions

![](./img/bias.jpg)

Let $n_l$ be the number of neurons in layer $l$.

**Weight matrix**

$$
W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_{l}}
$$

**Bias vector**

$$
b^{(l)} \in \mathbb{R}^{1 \times n_{l}}
$$

**Input activation**

$$
a^{(l-1)} \in \mathbb{R}^{1 \times n_{l-1}}
$$

**Output activation**

$$
z^{(l)}, a^{(l)} \in \mathbb{R}^{1 \times n_{l}}
$$

Each column of $W^{(l)}$ corresponds to a neuron in layer $l$:

$$
z_i^{(l)} = a^{(l-1)} \cdot w_{:,i}^{(l)} + b_i^{(l)}
$$

---

## 4. Vectorized Computation

Instead of computing neuron-by-neuron:

**Single neuron**

$$
z = x W + b
$$

**Entire layer**

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
$$

Vectorization allows computing all neurons in a layer simultaneously, which is efficient on modern hardware.

---

## 5. Depth vs Width

**Width**: number of neurons in a layer $n_l$
*More neurons → more capacity per layer*

**Depth**: number of layers $L$
*More layers → more hierarchical representations*

Intuition:

* Width → learn more features at the same level
* Depth → learn compositional or hierarchical features

---

## 6. Putting Everything Together


![](./img/numofcoefficients.jpg)

A neural network is defined by:

**Parameters**

$$
{ W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, \dots, W^{(L)}, b^{(L)} }
$$

**Forward Computation**

$$
a^{(l)} = g^{(l)}(a^{(l-1)} W^{(l)} + b^{(l)}), \quad l = 1, 2, \dots, L
$$

**Final Output**

$$
\hat{y} = a^{(L)}
$$
