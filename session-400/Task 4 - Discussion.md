# Task 4 Follow-Up: Bag of Words — Turning Language into Feature Vectors

## Introduction: From Language to Numbers

Machine learning models do not process text directly. They require structured numerical input:

$$
x \in \mathbb{R}^{1 \times d}
$$

So before any model (Logistic Regression, SVM, Neural Networks), we must solve a basic problem:

# How do we convert text into features?

This is the core of feature engineering in NLP.

---

## A Very Brief Reminder from Task 3

Earlier, we briefly saw another approach:

words were mapped to pretrained embeddings and averaged:

$$
v_{\text{document}} = \frac{1}{n}\sum_{i=1}^{n} v_i
$$

This introduces semantic information.

But it still collapses structure and order.

Now we move to a more fundamental and widely used baseline:

# Bag of Words (BoW)

---

# Bag of Words: The Core Idea

Bag of Words ignores meaning and structure.

It only asks:

## “Which words appear, and how many times?”

---

## Step 1: Vocabulary Construction

Given a dataset, we build a vocabulary:

$$
V = {w_1, w_2, \dots, w_d}
$$

Example:

$$
V =
{
\text{Shanghai},
\text{noodles},
\text{food},
\text{great},
\text{traffic},
\text{bad}
}
$$

So each word becomes one feature dimension.

---

## Step 2: Count Words in a Document

Consider the sentence:

```txt id="kq7d8p"
"Shanghai noodles are great. Shanghai food is great. Shanghai traffic is bad."
```

Now we compute counts:

### Word counts:

$$
\text{Shanghai} = 3
$$

$$
\text{noodles} = 1
$$

$$
\text{food} = 1
$$

$$
\text{great} = 2
$$

$$
\text{traffic} = 1
$$

$$
\text{bad} = 1
$$

---

## Step 3: Bag of Words Vector

The document becomes:

$$
x = [3,\ 1,\ 1,\ 2,\ 1,\ 1]
$$

Now text is a fixed-size vector.

---

## Key Property: Order Is Removed

Two sentences:

```txt id="2h4xv1"
"Shanghai food is great, traffic is bad"
```

```txt id="9k3m1q"
"Traffic is bad, Shanghai food is great"
```

Both produce the same vector:

$$
x = [1,\ 0,\ 1,\ 1,\ 1,\ 1]
$$

So:

$$
\text{order information is lost}
$$

---

## Mathematical Definition

For each word $w_j$:

$$
x_j = \text{count}(w_j \text{ in document})
$$

So:

$$
x = [x_1, x_2, \dots, x_d]
$$

Where:

$$
x \in \mathbb{R}^{1 \times d}
$$

---

## Why This Works at All

Even without understanding meaning:

Certain words correlate strongly with labels.

### Positive words:

$$
{\text{great}, \text{amazing}, \text{excellent}}
$$

### Negative words:

$$
{\text{bad}, \text{terrible}, \text{boring}}
$$

So classification can work using statistics alone.

---

## From Features to Model

Once we have:

$$
x \in \mathbb{R}^{1 \times d}
$$

We can use:

### Logistic Regression:

$$
z = xW + b
$$

$$
\hat{y} = \sigma(z)
$$

Expanded:

$$
z = \sum_{j=1}^{d} x_j w_j + b
$$

Each word contributes independently.

---

## Problem 1: No Word Importance Control

Frequent words may dominate even if uninformative.

Example:

“Shanghai” appears everywhere → may not help classification.

---

## Improvement 1: Term Frequency (TF)

Normalize within a document:

$$
\text{TF}(w,d)=\frac{\text{count}(w,d)}{\text{total words in } d}
$$

This removes document length bias.

---

## Improvement 2: Document Frequency (DF)

Now we look globally.

$$
df(w) = \text{number of documents containing } w
$$

Example:

If “Shanghai” appears in many documents:

$$
df(\text{Shanghai}) \uparrow
$$

If “terrible” appears in fewer:

$$
df(\text{terrible}) \downarrow
$$

---

## Improvement 3: Inverse Document Frequency (IDF)

We penalize common words:

$$
\text{IDF}(w)=\log\left(\frac{N}{1+df(w)}\right)
$$

Where:

* $N$ = total documents
* $df(w)$ = document frequency

---

## Final Improvement: TF-IDF

Combine both:

$$
\text{TF-IDF}(w,d)=\text{TF}(w,d)\cdot \text{IDF}(w)
$$

---

## Interpretation

A word is important if:

### 1. It appears often in the document

### 2. It is rare across documents

So TF-IDF balances:

$$
\text{local importance} \times \text{global rarity}
$$

---

## Key Limitation of BoW and TF-IDF

Even with improvements, fundamental issues remain:

### 1. No word order

$$
\text{“not good”} \approx \text{“good”}
$$

---

### 2. No semantic similarity

$$
\text{great} \not\approx \text{excellent}
$$

---

### 3. Sparse high-dimensional vectors

If vocabulary size is:

$$
d = 50000
$$

Then most entries in:

$$
x \in \mathbb{R}^{1 \times d}
$$

are zero.

---

## Why We Still Study It

Because Bag of Words introduces core ideas:

### Feature extraction:

$$
\text{Text} \rightarrow \text{Vector}
$$

### Linear modeling:

$$
xW + b
$$

### Interpretability:

We can directly inspect word weights.

---

## Later Progression

As limitations became clear, NLP evolved:

$$
\text{BoW/TF-IDF}
\rightarrow
\text{Embeddings}
\rightarrow
\text{RNN / LSTM}
\rightarrow
\text{Transformers}
$$

Each step improves representation power.
