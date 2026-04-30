# L1/L2 Regularization — Geometry and Effect

---

## 1. Goal

We study regularization from a **geometric perspective**.

This reveals why L1 and L2 produce different solutions.

---

## 2. Constrained Form

Regularized optimization is equivalent to a constrained problem.


#### L2 Constraint

$$
\sum_{j=1}^{d} W_j^2 \leq c
$$

This defines an **Euclidean ball**.


#### L1 Constraint

$$
\sum_{j=1}^{d} \|W_j\| \leq c
$$

This defines a **diamond-shaped polytope**.

---

## 3. Geometric View of Learning

We minimize loss subject to a constraint.

Solution occurs where:

> loss contour first touches constraint boundary

---

## 4. L2 Geometry

L2 constraint:

* smooth boundary
* no sharp corners

#### Effect

* solution is typically dense
* weights are distributed smoothly
* no preference for axes

---

## 5. L1 Geometry

L1 constraint:

* sharp corners
* aligned with coordinate axes


#### Effect

At intersection points:

* solutions often hit corners
* many coordinates become zero

This induces **sparsity naturally**
