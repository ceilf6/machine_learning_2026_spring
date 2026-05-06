# Deep Dive into Softmax 

## Turning Neural Network Outputs into Probabilities

![](./img/sm2.jpg)

---

## 1. Why Softmax Is Needed

In a multiclass neural network, the final layer often produces raw scores:

$$
z = [z_1, z_2, \dots, z_K]
$$

These scores are called:

$$\text{logits}$$

---

## Example

Suppose a 3-class classifier outputs:

$$
z = [2, 1, 0]
$$

This means:

* Class 1 score = 2
* Class 2 score = 1
* Class 3 score = 0

---

## Problem

These are not probabilities because:

##### They do not sum to 1

$$
2 + 1 + 0 = 3
$$

##### They can be negative

Example:

$$
z = [-1, 3, 0]
$$

##### They are just relative scores

Neural networks naturally output preferences, not probabilities.

---

## 2. Softmax Formula

Softmax converts logits into probabilities:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

---

## Goal

Transform:

$$
z = [2,1,0]
$$

into:

$$
\hat{y} = [0.665, 0.245, 0.090]
$$

---

## Probability Rules

After softmax:

$$
0 \leq \hat{y}_i \leq 1
$$

$$
\sum_{i=1}^{K}\hat{y}_i = 1
$$

---

## 3. Step-by-Step Example 1 — Positive Values

---

## Logits

$$
z = [2,1,0]
$$

---

## Step 1: Exponentiate

$$
e^2 \approx 7.39
$$

$$
e^1 \approx 2.72
$$

$$
e^0 = 1
$$

---

## Step 2: Sum

$$
7.39 + 2.72 + 1 = 11.11
$$

---

## Step 3: Normalize

$$
\hat{y}_1 = \frac{7.39}{11.11} \approx 0.665
$$

$$
\hat{y}_2 = \frac{2.72}{11.11} \approx 0.245
$$

$$
\hat{y}_3 = \frac{1}{11.11} \approx 0.090
$$

---

## Final Result

$$
\hat{y} \approx [0.665, 0.245, 0.090]
$$

---

### Interpretation

Class 1 has the highest probability because it had the highest logit.

---

## 4. Step-by-Step Example 2 — Including Negative Values

---

## Logits

$$
z = [-1,0,1]
$$

---

## Step 1: Exponentiate

$$
e^{-1} \approx 0.368
$$

$$
e^0 = 1
$$

$$
e^1 \approx 2.72
$$

---

## Step 2: Sum

$$
0.368 + 1 + 2.72 = 4.088
$$

---

## Step 3: Normalize

$$
\hat{y}_1 = \frac{0.368}{4.088} \approx 0.090
$$

$$
\hat{y}_2 = \frac{1}{4.088} \approx 0.245
$$

$$
\hat{y}_3 = \frac{2.72}{4.088} \approx 0.665
$$

---

## Final Result

$$
\hat{y} \approx [0.090, 0.245, 0.665]
$$

---

### Key Insight

Negative logits are completely valid.
Softmax only cares about relative differences.

---

## 5. Example 3 — Large Score Gap

---

## Logits

$$
z = [10,2,-3]
$$

---

## Approximate Exponentials

$$
e^{10} \gg e^2 \gg e^{-3}
$$

---

## Softmax Output

$$
\hat{y} \approx [0.9996, 0.0003, 0.0000]
$$

---

### Interpretation

A much larger logit creates near certainty.

---

# 6. Why Exponentials?


![](./img/sm1.jpg)

Exponentials do two jobs:

---

## A. Remove Negatives

Since:

$$
e^x > 0
$$

All values become positive.

---

## B. Amplify Differences

Difference between logits:

$$
2 - 1 = 1
$$

But after exponentiation:

$$
e^2 / e^1 = e
$$

---

### Meaning


> Softmax magnifies stronger preferences.

---

## 7. Important Property — Relative Scores Matter More Than Absolute Scores

Compare:

$$
[2,1,0]
$$

and:

$$
[102,101,100]
$$

Softmax outputs are identical.


### Why?

Because adding the same constant changes nothing:

$$
\text{Softmax}(z) = \text{Softmax}(z + c)
$$


### Proof Idea

$$
\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}=
\frac{e^c e^{z_i}}{e^c \sum_j e^{z_j}}
$$

The constant cancels.


### Core Meaning

> Softmax depends on differences, not absolute scale shifts.

---

## 8. The Numerical Stability Trick — Subtract Max

This is one of the most practical softmax tricks.

---

## Problem

If:

$$
z = [1000,999,998]
$$

Then:

$$
e^{1000}
$$

is dangerously large.

---

## Solution

Subtract the maximum logit first:

$$
z' = z - \max(z)
$$

So:

$$
[1000,999,998] \rightarrow [0,-1,-2]
$$

---

## Stable Softmax

$$
\hat{y}_i =
\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}
$$

---

## New Exponentials

$$
e^0 = 1
$$

$$
e^{-1} \approx 0.368
$$

$$
e^{-2} \approx 0.135
$$

---

## Much Safer

Subtracting max prevents overflow without changing probabilities.

---

## 9. NumPy Implementation

## Basic Version

```python
import numpy as np

z = np.array([2.0, 1.0, 0.0])

exp_z = np.exp(z)
softmax = exp_z / np.sum(exp_z)

print("Softmax:", softmax)
print("Sum:", np.sum(softmax))
```


---

## 10. Stable NumPy Version (Recommended)

```python
import numpy as np

def softmax(z):
    z = z - np.max(z)      # stability trick
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

z1 = np.array([2.0, 1.0, 0.0])
z2 = np.array([-1.0, 0.0, 1.0])
z3 = np.array([1000.0, 999.0, 998.0])

print("z1:", softmax(z1))
print("z2:", softmax(z2))
print("z3:", softmax(z3))
```


---

### Important Observation

```text id="v3i8b4"
[1000,999,998] behaves exactly like [2,1,0]
```

Because relative gaps are the same.

---

## 11. Softmax vs Argmax

Softmax gives probabilities:

$$
[0.665, 0.245, 0.090]
$$

---

## Argmax gives final class

$$
\text{argmax}(z) = 1
$$

---

## Difference

### Softmax

```text id="6j8j3k"
“How confident is each class?”
```

---

### Argmax

```text id="0o8n7p"
“Which class wins?”
```

---

## Example

$$
[0.40,0.35,0.25]
$$

Argmax still picks class 1, but confidence is weak.

---

### Practical Rule

```text id="x38dsa"
Softmax for probability distribution
Argmax for final prediction
```

---

## 12. Softmax in Neural Networks

In standard multiclass classification:


Input → Hidden Layers → Final Linear Layer → Logits → Softmax


---

## Example

For digit recognition:

$$
z \in \mathbb{R}^{1 \times 10}
$$

Softmax converts 10 logits into 10 probabilities.

---

## 13. Softmax in LLMs and Transformers (Optional)

Softmax is also central in large language models.

---

## Next-Token Prediction

A transformer outputs vocabulary logits:

$$
z \in \mathbb{R}^{1 \times V}
$$

where $V$ is vocabulary size.

Softmax converts this into:

$$
P(\text{next token})
$$

---

## Example

```text id="1xq2kj"
"The cat sat on the ___"
```

Possible logits:

* mat: 8.2
* floor: 6.1
* moon: 1.3

Softmax converts these into token probabilities.

---

## Attention Mechanism

Inside self-attention:

$$
\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

### Meaning

Softmax turns similarity scores into attention weights.
