# Early Stopping

---

## 1. Motivation: Overfitting During Training

When training a model, we typically observe:

* training loss decreases monotonically
* validation loss decreases first, then increases

This indicates that the model begins to overfit the training data after a certain point.

Early stopping is a method to select the training iteration that best generalizes.

---

## 2. Core Idea

We monitor performance on a validation set during training.

Let:

$$
\mathcal{L}_{train}(t)
$$

$$
\mathcal{L}_{val}(t)
$$

be training and validation loss at epoch $t$.

We aim to find:

$$
t^* = \arg\min_t \mathcal{L}_{val}(t)
$$

Instead of training until convergence, we stop near $t^*$.

---

## 3. Training Dynamics

Typical behavior:

* Early stage: both $\mathcal{L}_{train}$ and $\mathcal{L}_{val}$ decrease
* Middle stage: validation loss reaches minimum
* Late stage: training loss continues decreasing, validation loss increases

This divergence indicates overfitting.

---

## 4. Stopping Criterion

A practical stopping rule uses patience.

We define:

* best validation loss: $\mathcal{L}_{best}$
* patience: $k$

Algorithm:

* if $\mathcal{L}_{val}$ improves, reset counter
* otherwise increase counter
* stop when counter exceeds $k$

---

## 5. Algorithm (Conceptual Form)

At each epoch $t$:

1. train model
2. compute $\mathcal{L}_{val}(t)$
3. update best loss if improved
4. if no improvement for $k$ steps, stop

This approximates:

$$
\min_t \mathcal{L}_{val}(t)
$$

without explicitly training all epochs.

---

## 6. Why It Works

Early stopping acts as implicit regularization.

Key effects:

* limits model complexity
* prevents over-optimization on training data
* selects parameters before overfitting region

From an optimization perspective:

* later training epochs often correspond to sharper minima
* earlier stopping tends to select flatter regions

---

## 7. Relation to Model Capacity

Even without changing architecture, training time controls effective capacity.

Long training:

* higher effective capacity
* stronger fitting of noise

Shorter training:

* reduced effective capacity
* smoother function

Thus, early stopping is equivalent to controlling model complexity through optimization duration.
