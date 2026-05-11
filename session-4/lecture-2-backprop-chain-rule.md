# Backpropagation — The Core: Chain Rule


![](./img/backpropagation_main2.gif)



---

## 1. Motivation: Why the Chain Rule?

After the forward pass, we have a loss:

$$
\mathcal{L} = \mathcal{L}(\hat{y}, y)
$$

Suppose the network is composed of multiple layers:

$$
\hat{y} = f^{(L)}(f^{(L-1)}(\dots f^{(1)}(x) \dots ))
$$

We want gradients with respect to **all parameters**:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}}
$$

Direct differentiation is cumbersome. The **chain rule** allows us to break complex derivatives into **simpler, local derivatives**.

---

## 2. Single-Variable Chain Rule

If a function is nested:

$$
y = f(g(x))
$$

Then:

$$
\underbrace{\frac{dy}{dx}}_{\text{total derivative}} = \underbrace{\frac{dy}{dg}}_{\text{outer derivative}} \cdot \underbrace{\frac{dg}{dx}}_{\text{inner derivative}}
$$

Example:

$$
y = (3x + 1)^2
$$

$$
u = 3x + 1
$$

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x + 1)
$$

**Takeaway:** The chain rule decomposes a derivative into **local contributions**.

---

## 3. Multi-Variable Chain Rule

For a function of multiple variables:

$$
z = f(x, y), \quad x = g(t), \quad y = h(t)
$$

The derivative of $z$ w.r.t $t$ is:

$$
\underbrace{\frac{dz}{dt}}_{\text{total derivative}} = \underbrace{\frac{\partial z}{\partial x}}_{\text{path through } x} \cdot \underbrace{\frac{dx}{dt}}_{\text{rate of } x} + \underbrace{\frac{\partial z}{\partial y}}_{\text{path through } y} \cdot \underbrace{\frac{dy}{dt}}_{\text{rate of } y}
$$

**Interpretation:** If multiple paths influence $z$, each path contributes **additively** to the total derivative.

---

## 4. Applying to Neural Networks

A neural network is a **chain of functions**:

$$
\mathcal{L} \leftarrow \hat{y} \leftarrow z^{(L)} \leftarrow a^{(L-1)} \leftarrow \dots \leftarrow a^{(1)} \leftarrow z^{(1)} \leftarrow x
$$

* Each layer receives a **gradient signal** from the next layer
* Each layer multiplies this signal by its **local derivative**

Formally, for a scalar loss $\mathcal{L}$:

$$
\boxed{\underbrace{\frac{\partial \mathcal{L}}{\partial z^{(l)}}}_{\text{error signal at layer } l} = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{(l+1)}}}_{\text{error from next layer}} \cdot \underbrace{\frac{\partial z^{(l+1)}}{\partial z^{(l)}}}_{\text{local derivative}}}
$$

**Interpretation:** Gradients **flow backward**, layer by layer.

---

## 5. Local Gradients

Each node in the computation graph only needs **local derivatives**:

* Linear layer: $z = a W + b$

$$
\frac{\partial z}{\partial W} = a^T, \quad \frac{\partial z}{\partial a} = W^T
$$

* Activation layer: $a = g(z)$

$$
\frac{\partial a}{\partial z} = g'(z)
$$

By combining local derivatives using the chain rule, we get the **full gradient**.

---

## 6. Simple Example: Two-Layer Chain

Suppose:

$$
\mathcal{L} = (a \cdot b + c)^2
$$

Forward pass:

$$
u = a \cdot b + c
$$

$$
\mathcal{L} = u^2
$$

Backward pass (chain rule):

$$
\underbrace{\frac{\partial \mathcal{L}}{\partial a}}_{\text{gradient w.r.t. } a} = \underbrace{\frac{\partial \mathcal{L}}{\partial u}}_{\text{gradient w.r.t. } u} \cdot \underbrace{\frac{\partial u}{\partial a}}_{\text{local derivative}} = 2u \cdot b
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = 2u \cdot a
$$

$$
\frac{\partial \mathcal{L}}{\partial c} = 2u \cdot 1 = 2u
$$

**Observation:** Each derivative is computed **locally** and combined **systematically**.

---

## 7. Key Takeaways

1. **Backpropagation = repeated application of the chain rule**
2. **Gradients flow backward** through layers
3. **Each node computes local derivatives**, contributing to upstream gradients
4. For nodes with multiple outgoing paths, **sum the contributions**
5. This decomposition allows **efficient computation** in deep networks
