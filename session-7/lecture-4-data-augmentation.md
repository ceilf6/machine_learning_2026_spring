# Data Augmentation

---

## 1. Motivation: Data is Limited

Machine learning models often suffer from limited training data.

When data is small:

* models overfit easily
* learned representations are fragile
* generalization is weak

Data augmentation addresses this by artificially increasing dataset diversity.

---

## 2. Core Idea

Instead of collecting new data, we apply transformations:

$$
x \rightarrow T(x)
$$

where:

* $x$ is an original sample
* $T(\cdot)$ is a transformation function
* $T(x)$ is a valid training sample

Key requirement:

* label must remain unchanged or consistently transformable

---

## 3. Image Data Augmentation

For vision tasks, common transformations include:

* geometric transformations (rotation, crop, flip)
* color transformations (brightness, contrast, jitter)
* noise injection

Formally:

$$
x' = T_{image}(x)
$$

Examples:

* horizontal flip preserves object identity
* small rotation does not change class label
* Gaussian noise improves robustness

These transformations enforce invariance to natural distortions.

---

## 4. NLP Data Augmentation

For text, augmentation is more constrained because structure matters.

### 4.1 Synonym Replacement

Replace words with semantically similar ones:

$$
x = (w_1, w_2, ..., w_n)
$$

$$
x' = (w_1, ..., \text{syn}(w_i), ..., w_n)
$$

Goal:

* preserve meaning
* increase lexical diversity

---

### 4.2 Back Translation

A sentence is translated to another language and back:

$$
x \rightarrow x_{fr} \rightarrow x'
$$

This preserves semantics while changing surface form.

---

### 4.3 Random Deletion and Insertion

* randomly remove non-critical words
* insert neutral or context-preserving tokens

These methods simulate noisy natural language.

---

## 5. Why It Improves Generalization

Data augmentation acts as:

* regularization via input noise
* expansion of effective dataset size
* constraint on learned invariances

It reduces overfitting by forcing the model to learn:

* structure, not memorization
* invariant features, not surface patterns

---

## 6. Summary

Data augmentation is a general principle across domains:

* images: geometric and photometric transforms
* text: semantic-preserving rewrites

Core idea:

$$
\text{increase data diversity without changing labels}
$$

This improves robustness and generalization across almost all machine learning tasks.
