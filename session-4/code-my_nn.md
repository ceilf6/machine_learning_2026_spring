# Neural Networks from Scratch (Baseline)

Session 4, Session 5 and Session 7 form a miniseries of NN from scratch.

## Goal of this session

In this session you implement a **fully working neural network training loop** from scratch using NumPy:

- A minimal layer system (`Layer`, `Dense`, `ReLU`)
- A numerically stable `softmax + cross-entropy` loss
- Forward pass + backpropagation
- Full-batch **Gradient Descent** (GD)

This file is the **baseline** for the next two sessions:

- Session 5 will refactor *“how parameters are updated”* into optimizer classes (GD / SGD / Adam)
- Session 7 will add regularization (Dropout, L1/L2)

In Session 5 and Session 7, you will also see a more explicit, PyTorch-like separation:

- `Model`: runs the forward pass and knows which layers have parameters
- `CrossEntropyLoss`: produces a scalar loss and the gradient w.r.t. logits
- `Optimizer`: applies an update rule (`GD`, `SGD`, `Adam`)

## What you build (mental model)

The network is a list of layers:

- `Dense`: affine transform `XW + b`
- `ReLU`: non-linearity

Training follows the classic pattern:

1. Forward pass: produce logits
2. Loss: softmax cross-entropy
3. Backward pass: propagate gradients backwards
4. Parameter update: GD inside `Dense.backward()`

<details><summary>❓ What are “logits” and why do we compute them before softmax?</summary>

Logits are unnormalized class scores. Softmax turns logits into probabilities, and cross-entropy compares those probabilities to labels. Keeping logits until the loss is both standard and more numerically stable.

</details><br>

## The “one batch” picture (end-to-end)

Think of training as repeatedly doing the same thing on a batch. In Session 4 we use the *full dataset as one batch*.

```python
activations = forward(network, X)
logits = activations[-1]

loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

grad_output = grad_logits
for i in range(len(network))[::-1]:
    grad_output = network[i].backward(grad_output)
```

<details><summary>❓ What is the chain rule and why is it essential for backpropagation?</summary>

The chain rule computes derivatives of compositions. If `z=f(y)` and `y=g(x)`, then `dz/dx = (dz/dy) * (dy/dx)`. Backprop uses this idea to pass a gradient from the loss backward through each layer so every parameter gets an update direction.

</details><br>

## Key snippet: `Dense.backward()` does TWO jobs (Session 4 baseline)

This is the most important baseline idea for the progression:

- **Backprop**: compute `dW`, `db`, `dX`
- **Optimization**: apply an update rule (here: plain GD) to change `W` and `b`

```python
grad_weights = self.input.T @ grad_output
grad_biases = np.sum(grad_output, axis=0)
grad_input = grad_output @ self.weights.T

self.weights = self.weights - self.learning_rate * grad_weights
self.biases = self.biases - self.learning_rate * grad_biases
```

## How this maps to a PyTorch training step

In Session 4 we don’t yet have explicit `model`, `criterion`, `optimizer` objects. But the *roles* already exist.

- `forward(network, X)` is like `logits = model(X)`
- `softmax_crossentropy_with_logits(logits, y)` is like `loss = criterion(logits, y)`
- The reverse loop over layers is like `loss.backward()`
- The parameter update inside `Dense.backward()` is like `optimizer.step()`

In Session 5/6 you will see the same pattern written explicitly:

```python
logits = model(X_batch)
loss = criterion(logits, y_batch, model)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```


## Key snippet: stable softmax + cross-entropy

Two practical details matter:

- Subtract `max(logits)` before `exp` (numerical stability)
- Add a small epsilon inside `log`

```python
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
loss = -np.sum(one_hot * np.log(probs + 1e-9)) / batch_size
grad = (probs - one_hot) / batch_size
```

<details><summary>❓ What does “numerically stable softmax” mean?</summary>

We compute `exp(logits - max(logits))` to prevent overflow. Subtracting the maximum shifts all logits without changing the resulting probabilities.

</details><br>

<details><summary>❓ Why do we add a small epsilon inside `log(probs + 1e-9)`?</summary>

To avoid `log(0)` which would produce `-inf` and break training when probabilities become extremely small.

</details><br>


## Checklist: what must be true (debugging invariants)

- **Shapes line up**
  - `X`: `(batch, 784)`
  - logits: `(batch, 10)`
  - gradients flow backward with matching shapes
- **Loss decreases** (not necessarily every epoch, but trend)
- **Accuracy is above random** (`>10%` on MNIST)
- **Train vs Val accuracy**
  - training accuracy is usually higher; a large gap can mean overfitting


<details><summary>❓ What does `forward(network, X)` return in this session?</summary>

It returns a list of activations (the output of each layer).

For backprop, each layer stores what it needs during `forward(...)` (for example `Dense.input`), so the training loop only passes gradients backward.

</details><br>


<details><summary>❓ Why is the last layer `Dense(..., 10)` without an explicit softmax layer?</summary>

Softmax is applied inside `softmax_crossentropy_with_logits`. The network outputs logits, and the loss turns them into probabilities internally.

</details><br>

<details><summary>❓ Why do we use He-like initialization for `Dense` weights?</summary>

It helps keep activation/gradient magnitudes in a reasonable range as depth increases, reducing vanishing/exploding behavior early in training.

</details><br>

## What to take away

- You can build a working NN with only matrix multiplications and a few small functions.
- Forward/backward are just repeated application of the chain rule.
- In this baseline, **the layer updates its own parameters**.

Next session (Session 5): you’ll keep the layer math but move updates into **optimizer objects**.
