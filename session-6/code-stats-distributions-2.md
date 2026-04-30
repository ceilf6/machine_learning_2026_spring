# Binomial Distribution, Central Limit Theorem, and Bernoulli Processes 

# Setup

Let's begin by importing the libraries we will use.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

plt.rcParams["figure.figsize"] = [10,6]
```

---

# Part 1 — Bernoulli Distribution

Before studying the binomial distribution, we start with its simplest building block.

A **Bernoulli random variable** represents a single experiment with two possible outcomes:

* success (1)
* failure (0)

Examples:

| Example             | Success         |
| ------------------- | --------------- |
| Coin flip           | heads           |
| Neural unit dropout | unit dropped    |
| RL action sampling  | action selected |

The Bernoulli distribution has one parameter:

$$
p = P(X=1)
$$

---

## Exercise 1 — Sampling Bernoulli Variables

Generate 1000 Bernoulli trials with probability (p=0.7).

```python
p = 0.7
n_samples = 1000

# Your task: generate Bernoulli samples
samples = ______YOUR_CODE_HERE_________

plt.hist(samples, bins=2, alpha=0.7)
plt.xticks([0,1])
plt.title("Bernoulli Samples")
plt.show()

print("Mean:", np.mean(samples))
```

<details>
<summary>Answer</summary>

```python
samples = np.random.binomial(1, p, n_samples)
```

</details>

---

## Exercise 2 — Exploring Different Probabilities

Let's observe how Bernoulli distributions change when (p) changes.

```python
probs = [0.1,0.3,0.5,0.7,0.9]

for p in probs:

    samples = ______YOUR_CODE_HERE_________

    plt.hist(samples, bins=2, alpha=0.5, label=f"p={p}")

    plt.legend()
    plt.title("Bernoulli distributions")
    plt.show()
```

<details>
<summary>Answer</summary>

```python
samples = np.random.binomial(1,p,1000)
```

</details>

---

# Part 2 — Binomial Distribution

A **Binomial distribution** counts the number of successes in multiple Bernoulli trials.

If we perform:

* (n) independent trials
* success probability (p)

then

$$
X \sim Binomial(n,p)
$$

---

## Mathematical Definition

The probability of observing $k$ successes:

$$
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$

---

## Exercise 3 — Generating Binomial Samples

Generate samples from a binomial distribution with:

* (n=20)
* (p=0.5)

```python
n = 20
p = 0.5
num_samples = 5000

samples = ______YOUR_CODE_HERE_________

plt.hist(samples,bins=range(n+2),density=True,alpha=0.7)
plt.title("Binomial Distribution")
plt.xlabel("Number of successes")
plt.ylabel("Probability")
plt.show()
```

<details>
<summary>Answer</summary>

```python
samples = np.random.binomial(n,p,num_samples)
```

</details>

---

## Exercise 4 — Mean and Variance

The binomial distribution has:

$$
E[X]=np
$$

$$
Var(X)=np(1-p)
$$

Verify this experimentally.

```python
sample_mean = ______YOUR_CODE_HERE_________
sample_var = ______YOUR_CODE_HERE_________

print("Sample mean:", sample_mean)
print("Expected mean:", n*p)

print("Sample variance:", sample_var)
print("Expected variance:", n*p*(1-p))
```

<details>
<summary>Answer</summary>

```python
sample_mean = np.mean(samples)
sample_var = np.var(samples)
```

</details>

---

# Part 3 — Visualizing the Effect of Parameters

## Exercise 5 — Changing Probability

Observe how the shape of the distribution changes.

```python
n = 40
ps = [0.1,0.3,0.5,0.7,0.9]

plt.figure(figsize=(10,6))

for p in ps:

    samples = ______YOUR_CODE_HERE_________

    plt.hist(samples,bins=range(n+2),density=True,alpha=0.5,label=f"p={p}")

plt.legend()
plt.title("Effect of Probability on Binomial Distribution")
plt.show()
```

<details>
<summary>Answer</summary>

```python
samples = np.random.binomial(n,p,3000)
```

</details>

---

## Exercise 6 — Changing Number of Trials

```python
p = 0.5
ns = [10,20,50,100]

plt.figure(figsize=(10,6))

for n in ns:

    samples = ______YOUR_CODE_HERE_________

    plt.hist(samples,bins=range(n+2),density=True,alpha=0.5,label=f"n={n}")

plt.legend()
plt.title("Effect of Number of Trials")
plt.show()
```

<details>
<summary>Answer</summary>

```python
samples = np.random.binomial(n,p,3000)
```

</details>

---

# Part 4 — Normal Approximation of Binomial

For large $n$, the binomial distribution approaches a normal distribution.

$$
\mu = np
$$

$$
\sigma = \sqrt{np(1-p)}
$$

---

## Exercise 7 — Overlaying Normal Distribution

```python
n = 60
p = 0.5

samples = np.random.binomial(n,p,5000)

mu = ______YOUR_CODE_HERE_________
sigma = ______YOUR_CODE_HERE_________

x = np.linspace(0,n,1000)

pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

plt.hist(samples,bins=range(n+2),density=True,alpha=0.6)
plt.plot(x,pdf,"r-",linewidth=2)
plt.title("Binomial vs Normal Approximation")
plt.show()
```

<details>
<summary>Answer</summary>

```python
mu = n*p
sigma = np.sqrt(n*p*(1-p))
```

</details>

---

# Part 5 — Central Limit Theorem (CLT)

The **Central Limit Theorem** is one of the most important results in statistics.

It states:

> The sum (or average) of many independent random variables tends toward a normal distribution, regardless of the original distribution.

---

## Exercise 8 — CLT Using Uniform Distribution

```python
n_samples = 5000
n_variables = 40

data = np.random.uniform(0,1,(n_samples,n_variables))

sums = ______YOUR_CODE_HERE_________

plt.hist(sums,bins=40,density=True,alpha=0.7)
plt.title("CLT demonstration")
plt.show()
```

<details>
<summary>Answer</summary>

```python
sums = np.sum(data,axis=1)
```

</details>

---

## Exercise 9 — Increasing the Number of Variables

Observe how the distribution becomes more normal.

```python
n_samples = 5000
variables = [1,2,5,10,30]

plt.figure(figsize=(12,8))

for i,n_vars in enumerate(variables):

    data = np.random.uniform(0,1,(n_samples,n_vars))

    sums = ______YOUR_CODE_HERE_________

    plt.subplot(2,3,i+1)
    plt.hist(sums,bins=30,density=True)
    plt.title(f"{n_vars} variables")

plt.tight_layout()
plt.show()
```

<details>
<summary>Answer</summary>

```python
sums = np.sum(data,axis=1)
```

</details>

---

# Part 6 — Why CLT Matters in Deep Learning

Neural networks train using **mini-batch gradient descent**.

Instead of computing the gradient on the entire dataset:

$$
\nabla L = \frac{1}{N} \sum_{i=1}^{N} g_i
$$

we estimate it using a **mini-batch**:

$$
\hat{\nabla L} = \frac{1}{B} \sum_{i=1}^{B} g_i
$$

CLT explains why this works.

The average of random gradient samples tends toward a normal distribution centered at the true gradient.

---

## Exercise 10 — Simulating Gradient Estimates

```python
true_gradient = 2.0

all_gradients = np.random.normal(true_gradient,0.5,10000)

batch_size = 32

estimates = []

for i in range(200):

    batch = ______YOUR_CODE_HERE_________

    estimates.append(np.mean(batch))

plt.hist(estimates,bins=30,density=True)
plt.title("Mini-batch Gradient Estimates")
plt.show()

print("Mean estimate:", np.mean(estimates))
```

<details>
<summary>Answer</summary>

```python
batch = np.random.choice(all_gradients,batch_size)
```

</details>

---

# Part 7 — Batch Size and Gradient Variance

Smaller batches produce noisier gradients.

---

## Exercise 11 — Compare Batch Sizes

```python
true_gradient = 2.0

all_gradients = np.random.normal(true_gradient,0.5,10000)

batch_sizes = [2,8,32,128]

plt.figure(figsize=(12,8))

for i,batch_size in enumerate(batch_sizes):

    estimates = []

    for _ in range(200):

        batch = ______YOUR_CODE_HERE_________

        estimates.append(np.mean(batch))

    plt.subplot(2,2,i+1)
    plt.hist(estimates,bins=30,density=True)
    plt.title(f"Batch size {batch_size}")

plt.tight_layout()
plt.show()
```

<details>
<summary>Answer</summary>

```python
batch = np.random.choice(all_gradients,batch_size)
```

</details>

---

# Part 8 — Dropout as a Bernoulli Process

In neural networks, **dropout randomly disables neurons during training**.

Each neuron is kept with probability (p).

This is exactly a **Bernoulli random variable**.

---

## Exercise 12 — Simulating Dropout

```python
n_units = 20
dropout_rate = 0.3

mask = ______YOUR_CODE_HERE_________

print(mask)
print("Active neurons:", np.sum(mask))
```

<details>
<summary>Answer</summary>

```python
mask = np.random.binomial(1,1-dropout_rate,n_units)
```

</details>

---

## Exercise 13 — Visualizing Dropout

```python
n_units = 100
dropout_rate = 0.5

masks = []

for _ in range(200):

    mask = ______YOUR_CODE_HERE_________

    masks.append(np.sum(mask))

plt.hist(masks,bins=30)
plt.title("Number of Active Neurons After Dropout")
plt.show()
```

<details>
<summary>Answer</summary>

```python
mask = np.random.binomial(1,1-dropout_rate,n_units)
```

</details>

---

# Part 9 — Bernoulli Processes in Reinforcement Learning

In reinforcement learning, policies often output **probabilities of actions**.

For binary actions:

$$
P(action=1)=p
$$

Action sampling becomes a **Bernoulli draw**.

---

## Exercise 14 — Action Sampling

```python
action_prob = 0.7

actions = ______YOUR_CODE_HERE_________

unique,counts = np.unique(actions,return_counts=True)

print(dict(zip(unique,counts)))

plt.hist(actions,bins=2)
plt.xticks([0,1])
plt.title("Action Sampling")
plt.show()
```

<details>
<summary>Answer</summary>

```python
actions = np.random.binomial(1,action_prob,1000)
```

</details>

---

# Final Summary

This tutorial introduced several key ideas:

| Concept                | Application                          |
| ---------------------- | ------------------------------------ |
| Bernoulli distribution | binary outcomes                      |
| Binomial distribution  | repeated trials                      |
| Normal approximation   | large-sample behavior                |
| Central Limit Theorem  | explains normality in many processes |
| Mini-batch gradients   | stochastic gradient descent          |
| Dropout                | Bernoulli masking                    |
| RL action sampling     | probabilistic policies               |

These concepts are fundamental to understanding why many machine learning algorithms behave the way they do.
