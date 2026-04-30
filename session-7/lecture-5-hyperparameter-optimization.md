# Hyperparameter Optimization

---

## 1. Parameters vs Hyperparameters

A machine learning model contains two types of variables:

### Parameters

Parameters are learned from data during training.

Example:

$$
\theta = \{W, b\}
$$

They are optimized by minimizing a loss function:

$$
\theta^* = \arg\min_\theta \mathcal{L}(\theta)
$$

---

### Hyperparameters

Hyperparameters are not learned from data.

They are fixed before training.

Examples:

* learning rate $\eta$
* regularization strength $\lambda$
* number of layers
* batch size
* tree depth

They define the learning process itself, not the model output.

---

### Key Distinction

* parameters: learned
* hyperparameters: selected

Hyperparameters control how optimization happens, not what is optimized.

---

## 2. Motivation: Why Optimization is Needed

Model performance depends strongly on hyperparameters.

Different choices of:

* learning rate
* model depth
* regularization strength

can lead to:

* underfitting
* overfitting
* unstable training

Thus, hyperparameter selection is itself an optimization problem.

---

## 3. Problem Formulation

We define a validation objective:

$$
\mathcal{L}_{val}(\lambda)
$$

where $\lambda$ represents hyperparameters.

Goal:

$$
\lambda^* = \arg\min_\lambda \mathcal{L}_{val}(\lambda)
$$

Unlike parameters, this objective is:

* non-convex
* expensive to evaluate
* black-box

---

## 4. Grid Search

Grid search evaluates all combinations in a predefined set.

Example:

$$
\eta \in {0.1, 0.01}, \quad \lambda \in {0.1, 0.01}
$$

Total evaluations:

$$
2 \times 2 = 4
$$

### Properties

* exhaustive
* simple
* computationally expensive in high dimensions

---

## 5. Random Search

Random search samples hyperparameters from distributions.

Instead of full enumeration:

$$
\lambda \sim p(\lambda)
$$

Each trial evaluates a random configuration.

### Key Insight

Not all hyperparameters are equally important.

Random search allocates more trials to exploring important dimensions effectively.

---

## 6. Why Random Search Works Better

Consider two hyperparameters:

* $\eta$: highly sensitive
* $\lambda$: weakly sensitive

Grid search wastes computation by repeatedly exploring $\lambda$.

Random search instead:

* explores more diverse values of $\eta$
* avoids redundant coverage of insensitive dimensions

This improves the probability of finding good configurations under limited budget.

---

## 7. Hyperparameter Optimization as Black-Box Search

We only observe:

$$
\lambda \rightarrow \mathcal{L}_{val}
$$

We do not know:

* gradients
* curvature
* structure

Thus optimization methods include:

* grid search
* random search
* Bayesian optimization (later topic)

---

## 8. Practical Workflow

Typical workflow:

1. define search space
2. sample hyperparameters
3. train model
4. evaluate validation loss
5. repeat

Best configuration is selected by:

$$
\min \mathcal{L}_{val}
$$

---

## 9. Summary

Hyperparameter optimization is a second-level optimization problem where:

* parameters are learned
* hyperparameters are selected

Grid search is exhaustive but inefficient.

Random search is simple and often more effective in practice.

The core challenge is that hyperparameter spaces are:

* high-dimensional
* expensive to evaluate
* non-differentiable

