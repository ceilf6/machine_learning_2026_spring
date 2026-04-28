# Multiple Linear Regression


![](./img/ml.jpg)

---

# 1. From One Feature to Many

In simple linear regression, we use one feature:

$$
\hat{y} = w x + b
$$

In many real problems, one feature is not enough.

We often have multiple features:

* house size
* number of rooms
* location

So we extend the model.

---

# 2. Model Formulation

For $d$ features, the model becomes:

$$
\boxed{\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b}
$$

Each feature has its own weight.

We can write this in vector form using the **row-vector convention**:

$$
\boxed{\hat{y} = x W + b}
$$

where:

* $x \in \mathbb{R}^{1 \times d}$ (input row vector)
* $W \in \mathbb{R}^{d \times 1}$ (weight matrix)
* $b \in \mathbb{R}^{1 \times 1}$ (bias)

#### Interpretation

Each weight $w_j$ shows how much feature $x_j$ affects the prediction.

* positive $w_j$ → increases prediction
* negative $w_j$ → decreases prediction

The model is still **linear in parameters**, even with many features.

---

# 3. Dataset Representation

We have $n$ samples:

$$
\{(x^{(i)}, y^{(i)})\}_{i=1}^n
$$

Each $x^{(i)}$ is now a row vector:

$$
x^{(i)} = [x_1^{(i)}, x_2^{(i)}, \dots, x_d^{(i)}] \in \mathbb{R}^{1 \times d}
$$

The prediction is:

$$
\hat{y}^{(i)} = x^{(i)} W + b
$$

---

# 4. Loss Function

We still use **Mean Squared Error (MSE)**:

$$
\boxed{\mathcal{L}(W, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2}
$$


---

# 5. Gradients

We compute gradients with respect to all parameters.

For each weight $w_j$:

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
$$

In vector form:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W} = \frac{2}{n} \sum_{i=1}^n x^{(i)\mathsf{T}} (\hat{y}^{(i)} - y^{(i)})}
$$

For the bias:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})}
$$

We update all parameters using gradient descent.


Or, in matrix form (full batch; and be used for non-full batch as well):

$$
\boxed{W \leftarrow W - \eta
\frac{2}{n}
X^{\mathsf T}
(\hat{Y}-Y)}
$$

$$
\boxed{b \leftarrow b - \eta \frac{2}{n} \sum_{i=1}^{n}
(\hat{y}_i-y_i)}
$$


---

# 6. Batch Matrix Form (Optional)

We can write everything in a compact matrix form for efficient computation.

Following the **row-vector convention**, we stack $n$ samples as rows in the design matrix $X \in \mathbb{R}^{n \times d}$:

$$
X =
\begin{bmatrix}
x_1^{(1)} & x_2^{(1)} & \cdots & x_d^{(1)} \\
x_1^{(2)} & x_2^{(2)} & \cdots & x_d^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
x_1^{(n)} & x_2^{(n)} & \cdots & x_d^{(n)}
\end{bmatrix}
$$

Define the weight matrix $W \in \mathbb{R}^{d \times 1}$:

$$
W = 
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_d
\end{bmatrix}
$$

And the target vector $y \in \mathbb{R}^{n \times 1}$:

$$
y = 
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(n)}
\end{bmatrix}
$$

Now, all predictions can be written as:

$$
\hat{y} = XW + \mathbf{1}b
$$

where $\hat{y} \in \mathbb{R}^{n \times 1}$ is the vector of predictions for all samples, and $\mathbf{1} \in \mathbb{R}^{n \times 1}$ is a column vector of ones.

The Mean Squared Error (MSE) loss function in matrix form is:

$$
\mathcal{L}(W, b) = \frac{1}{n}(XW + \mathbf{1}b - y)^{\mathsf{T}}(XW + \mathbf{1}b - y)
$$

This is equivalent to:

$$
\mathcal{L}(W, b) = \frac{1}{n}\|XW + \mathbf{1}b - y\|_2^2
$$

where $\|\cdot\|_2$ denotes the Euclidean norm.

The gradients of the loss are:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{2}{n}X^{\mathsf{T}}(XW + \mathbf{1}b - y)
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n}\mathbf{1}^{\mathsf{T}}(XW + \mathbf{1}b - y)
$$

