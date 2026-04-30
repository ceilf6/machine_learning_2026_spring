# Bias–Variance Tradeoff

---

## 1. Goal

Even with correct evaluation, models can fail in different ways.

We need to understand **why errors happen**.

---

## 2. Two Sources of Error

Model error comes from two main sources:

* bias
* variance

---

## 3. High Bias (Underfitting)

A model with high bias is too simple.

It cannot capture the structure of the data.

---

## Symptoms

* high training error
* high validation error
* both are similar

---

## Interpretation

The model is **systematically wrong**.

It misses key patterns.

---

## Example

* linear model on nonlinear data
* shallow decision tree

---

## 4. High Variance (Overfitting)

A model with high variance is too sensitive to data.

It learns noise in the training set.

---

## Symptoms

* low training error
* high validation error
* large gap between them

---

## Interpretation

The model is **unstable across datasets**.

Small changes in data lead to large changes in predictions.

---

## Example

* very deep decision tree
* high-degree polynomial regression

---

## 5. Bias–Variance Tradeoff

Increasing model complexity:

* reduces bias
* increases variance

Decreasing model complexity:

* increases bias
* reduces variance

---

## Key Idea

There is a **sweet spot** between underfitting and overfitting.

---

## 6. Learning Curves

Learning curves plot:

* training error
* validation error
* vs dataset size

---

## Patterns

### High Bias

* both errors high
* curves close together

---

### High Variance

* training error low
* validation error high
* large gap

---

## 7. Model Selection Insight

Good models achieve:

* low bias
* controlled variance
