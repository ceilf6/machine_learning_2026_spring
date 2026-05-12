# Backpropagation — The Core: Chain Rule


![](./img/backpropagation_main2.gif)



---

## 1. Motivation

In a deep network, the loss depends on each parameter through a long chain of composed functions. Computing gradients directly is impractical. The **chain rule** lets us decompose these derivatives into **manageable, local pieces**.

---

## 2. Single-Variable Chain Rule

If a function is nested:

$$
y = f(g(x))
$$

Then:

$$
\underbrace{\frac{dy}{dx}}_{\text{total derivative}} = \underbrace{\frac{dy}{dg}}_{\text{outer derivative}} \, \underbrace{\frac{dg}{dx}}_{\text{inner derivative}}
$$

Example:

$$
y = (3x + 1)^2
$$

$$
u = 3x + 1
$$

$$
\frac{dy}{dx} = \frac{dy}{du} \, \frac{du}{dx} = 2u \cdot 3 = 6(3x + 1)
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
\underbrace{\frac{dz}{dt}}_{\text{total derivative}} = \underbrace{\frac{\partial z}{\partial x}}_{\text{path through } x} \, \underbrace{\frac{dx}{dt}}_{\text{rate of } x} + \underbrace{\frac{\partial z}{\partial y}}_{\text{path through } y} \, \underbrace{\frac{dy}{dt}}_{\text{rate of } y}
$$

**Interpretation:** If multiple paths influence $z$, each path contributes **additively** to the total derivative.

---

## 4. Applying to Neural Networks

A neural network is a **chain of functions**:

$$
\boxed{\mathcal{L} \longleftarrow \hat{y} \longleftarrow z^{(L)} \longleftarrow a^{(L-1)} \longleftarrow \dots \longleftarrow a^{(1)} \longleftarrow z^{(1)} \longleftarrow x}
$$

* Each layer receives a **gradient signal** from the next layer
* Each layer multiplies this signal by its **local derivative**

Formally, for a scalar loss $\mathcal{L}$:

$$
\boxed{\underbrace{\frac{\partial \mathcal{L}}{\partial z^{(l)}}}_{\text{error signal at layer } l} = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{(l+1)}}}_{\text{error from next layer}} \, \underbrace{\frac{\partial z^{(l+1)}}{\partial z^{(l)}}}_{\text{local derivative}}}
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

## 6. Key Takeaways

1. **Backpropagation = repeated application of the chain rule**
2. **Gradients flow backward** through layers
3. **Each node computes local derivatives**, contributing to upstream gradients
4. For nodes with multiple outgoing paths, **sum the contributions**
5. This decomposition allows **efficient computation** in deep networks
