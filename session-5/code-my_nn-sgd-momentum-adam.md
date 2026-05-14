# Neural Networks from Scratch — Session 5 (SGD + Momentum + Adam)

Session 4, Session 5 and Session 7 form a miniseries of NN from scratch.


## Where we are in the 3-session progression

- **Session 4**: you implemented the full forward + loss + backprop pipeline, and each `Dense` layer updated its own parameters.
- **Session 5** (this file): you keep the *same math*, but you refactor parameter updates into **optimizers**.
- **Session 7**: you will keep this Session 5 structure and add **regularization** (Dropout, L1/L2).

## Learning objectives (what you should be able to explain)

- **Why** Session 4’s code “worked” but was hard to extend.
- **What** an optimizer is: a reusable update rule + its own state.
- **How** mini-batches change the training loop (but not the gradients).
- **How** this maps to the familiar PyTorch rhythm:

`model(X) -> loss = criterion(...) -> zero_grad -> backward -> step`.


## What changes from Session 4

Session 4 worked, but it mixed responsibilities:

- The `Dense` layer computed gradients **and** updated parameters using a fixed update rule.

In this session you refactor the update rule into **optimizer objects**:

- **GD**: full-batch Gradient Descent
- **SGD**: mini-batch Stochastic Gradient Descent
- **Momentum**
- **Adam**: adaptive learning rate + momentum

## Key design idea: separate responsibilities (PyTorch mental model)

We keep the same math as Session 4 (forward, loss, backprop), but we separate the roles:

- **Layers** (`Dense`, `ReLU`):
  - Forward pass
  - Backward pass (compute gradients)
- **Training loop**:
  - Decides batching (full batch vs mini-batch)
  - Calls forward/backward on each batch
- **Optimizer**:
  - Applies an update rule (`GD`, `SGD`, `Momentum`, `Adam`) to parameters using stored gradients

In code, these roles are expressed explicitly as:

- `Model`: wraps a list of layers and exposes `parameters()`.
- `CrossEntropyLoss`: computes loss value and gradient w.r.t. logits.
- `Optimizer`: owns the update rule and (for Adam) the moving-average state.

This mirrors the idea of `loss.backward()` + `optimizer.step()`.

## The new mental model

Session 4 had this hidden coupling:

- **Backward pass** computed gradients
- **Layer** immediately applied an update

Session 5 separates the concerns:

- **Layers** compute and store gradients
- **Optimizers** update parameters
- **Training loop** decides *batching* and *when* to call `step()`


## Key snippet: `Dense.backward()` now stores gradients (no update)

```python
grad_weights = self.input.T @ grad_output
grad_biases = np.sum(grad_output, axis=0)
grad_input = grad_output @ self.weights.T

self.grad_weights = grad_weights
self.grad_biases = grad_biases

return grad_input
```


## Key snippet: `optimizer.step()` updates all `Dense` layers

All optimizers follow the same interface:

```python
logits = model(X_batch)
loss = criterion(logits, y_batch, model)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Inside `step`, loop through the parameter-bearing layers and update them.

<details><summary>❓ Why does `optimizer.zero_grad()` set gradients to zeros?</summary>

It makes the training step safer: gradients always exist, and each batch overwrites them with fresh values. This mirrors the “clear grads then compute grads” rhythm in PyTorch.

</details><br>


## Optimizers: what they change (intuitively)

- **GD (full-batch)**
  - One update per epoch using the whole dataset.
  - Smooth updates, but each update is expensive.
- **SGD (mini-batch)**
  - Many updates per epoch using random mini-batches.
  - Updates are noisy, but often reach a good solution faster.
- **Adam**
  - Adds moving averages of gradients (momentum-like).
  - Also rescales updates using a moving average of squared gradients (adaptive learning rates).

<details><summary>❓ Why can SGD/Adam sometimes generalize better than full-batch GD?</summary>

Mini-batch noise can act like a weak regularizer and may help avoid sharp minima. It’s not guaranteed, but it’s a common practical observation.

</details><br>

<details><summary>❓ If Adam is “better”, why learn GD and SGD at all?</summary>

Because GD/SGD are the simplest mental models. Adam is easiest to understand as “SGD + momentum-like smoothing + per-parameter scaling”.

</details><br>

## Mini-batches: one idea, two loops

You will see two different outer loops depending on whether you are doing full-batch GD or mini-batch (SGD/Adam):

```python
if batch_size:
    for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
        logits = model(X_batch)
        loss = criterion(logits, y_batch, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
else:
    logits = model(X_train)
    loss = criterion(logits, y_train, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

<details><summary>❓ Why is GD “full-batch” in this tutorial?</summary>

It’s the cleanest baseline: one update per epoch using the exact gradient over the dataset.

</details><br>


<details><summary>❓ What actually makes SGD “stochastic”?</summary>

The gradient is computed on a random mini-batch instead of the full dataset, so it’s a noisy estimate of the true gradient.

</details><br>



<details><summary>❓ What problem does Adam solve compared to SGD?</summary>

Adam uses moving averages of gradients (momentum-like) and rescales updates per-parameter (adaptive learning rates), often making training less sensitive to the learning rate.

</details><br>


## Key snippet: a minimal mini-batch iterator

```python
indices = np.random.permutation(num_samples)
for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    batch_idx = indices[start:end]
    yield X[batch_idx], y[batch_idx]
```


## Backprop “shape story” (what to keep in your head)

For a batch:

- Input `X`: `(batch, 784)`
- Hidden activations: `(batch, hidden)`
- Logits: `(batch, 10)`
- Gradient flows backward with the same shapes.

Predictions can be explained as:

`logits -> softmax -> probabilities -> argmax`.

Inside `Dense.backward()` you compute:

- `dW = X^T @ dY`
- `db = sum(dY, axis=0)`
- `dX = dY @ W^T`

Then the optimizer updates using the parameters it was constructed with.

## What to take away

- Backprop is unchanged; you are only refactoring *where* parameter updates happen.
- Mini-batches (SGD/Adam) give noisier but faster updates than full-batch GD.
- Adam is SGD + momentum + adaptive per-parameter learning rates.
- Training logs usually report both **train accuracy** and **validation accuracy** to reveal overfitting.

Next session (Session 7): keep this exact structure, then add **Dropout + L1/L2**.

