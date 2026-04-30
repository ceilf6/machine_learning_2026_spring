# Neural Networks from Scratch — Session 6 (Dropout, L1/L2)

## What changes from Session 5

Session 5 improved *optimization* (GD vs SGD vs Adam), but models can still overfit.

In this session you add **regularization**:

- **Dropout**: randomly remove activations during training
- **L1 / L2**: penalize large weights (and encourage sparsity for L1)

## Learning objectives (what you should be able to explain)

- **Why** overfitting can happen even when optimization is “good”.
- **How** Dropout changes the forward pass *during training only*.
- **How** L1/L2 change both the scalar loss and the gradient on `W`.

## Important engineering detail: train mode vs eval mode

Dropout must behave differently during:

- Training: apply a random mask
- Evaluation: do nothing

So we add `train_mode()` / `eval_mode()` to the base `Layer`, and we control modes explicitly via:

- `set_train_mode(network)`
- `set_eval_mode(network)`

Also: `forward()` should **not** silently change the mode. The caller decides.

<details><summary>❓ Why do we need train/eval mode at all?</summary>

Because some layers behave differently during training vs evaluation. Dropout is the simplest example.

</details><br>

In the Session 5 style, you can think of it as:

- `model.train()` before gradient steps
- `model.eval()` before validation/test metrics

## Key snippet: Dropout forward/backward (inverted dropout)

Dropout needs two ideas:

- Sample a binary mask during training
- Scale activations by `1/(1-p)` during training so the expected value stays the same

```python
if self.training:
    self.mask = binomial_mask / (1 - p)
    return input * self.mask
return input
```

<details><summary>❓ What happens if you forget to switch dropout off at evaluation?</summary>

Your validation/test accuracy will look artificially worse because you keep randomly deleting activations.

</details><br>

<details><summary>❓ Why is “inverted dropout” preferred?</summary>

Because evaluation becomes a no-op: the scaling happens only during training, keeping expected activations consistent.

</details><br>

Backward is just masking the gradient too:

```python
return grad_output * self.mask
```

## Key snippet: L1/L2 = (loss penalty) + (gradient term)

There are two places regularization must show up:

- Add a scalar penalty to the loss
- Add a gradient term to `dW`

```python
loss += l1 * sum(abs(W))
loss += l2 * sum(W**2)
```

```python
dW += l1 * sign(W)
dW += 2 * l2 * W
```

<details><summary>❓ Why does L1 encourage sparsity?</summary>

Its gradient adds a sign term that can push small weights to exactly zero more aggressively than L2.

</details><br>


## Putting it together: PyTorch-like training step (from scratch)

Even though the backprop is still implemented manually, the *usage pattern* is intentionally familiar:

```python
logits = model(X_batch)
loss = criterion(logits, y_batch, model)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

And because we now have dropout, we also make the mode explicit:

```python
model.train()  # dropout ON
... gradient steps ...

model.eval()   # dropout OFF
... validation metrics ...
```

In this session, the optimizer is constructed from `model.parameters()`, so `step()` updates exactly the parameter-bearing layers.

<details><summary>❓ With dropout/L2, should training accuracy always increase?</summary>

Not necessarily. Regularization can reduce training accuracy while improving validation accuracy. The goal is better generalization.

</details><br>

## What to take away

- Regularization is about controlling *generalization*, not just training loss.
- Dropout requires a clean and explicit **train/eval mode** system.
- L1/L2 can be implemented by:
  - Adding a penalty term to the loss (so the scalar loss matches what you optimize)
  - Adding the corresponding gradient term to `dW` inside `Dense.backward()`

This completes the three-session “NN from scratch” series.
