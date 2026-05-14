# Exploring tiny_glove Word Vectors


> [!INFO]
> **Task for students**
> - **Task 1**. Go to 
> - https://nlp.stanford.edu/projects/glove/
> - for downloading more glove embeddings
> - and then use the same code and see how things change or not change.
> - **Task 2**. Then go for word2vec embedding as well : 
> - https://www.kaggle.com/datasets/sugataghosh/google-word2vec
> - Or, https://huggingface.co/NeuML/word2vec
> - Or, maybe explore other token embeddings on huggingface, 
> - e.g. https://huggingface.co/BAAI/llm-embedder



---

# Step 1: Import Libraries

```python
import json
import numpy as np
from numpy.linalg import norm
```

---

# Step 2: Load tiny_glove

```python
with open("./datasets/tiny_glove.json", "r") as f:
    glove = json.load(f)
```

```python
print("Vocabulary size:", len(glove))

sample_words = list(glove.keys())[:20]

print("\nFirst 20 words:")
print(sample_words)
```

---

# Step 3: Inspect One Word Vector

```python
word = "king"

vector = np.array(glove[word])
```

```python
print("Word:", word)

print("Vector shape:", vector.shape)

print("First 10 values:")
print(vector[:10])
```

---

# Step 4: Build Utility Functions

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def get_vector(word):
    if word in glove:
        return np.array(glove[word])
    return None
```

---

# Step 5: Compare Basic Similarities

```python
pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("doctor", "nurse"),
    ("king", "apple"),
    ("teacher", "rich")
]
```

```python
for w1, w2 in pairs:
    v1 = get_vector(w1)
    v2 = get_vector(w2)

    if v1 is not None and v2 is not None:
        sim = cosine_similarity(v1, v2)

        print(f"{w1:10s} vs {w2:10s} -> {sim:.4f}")
    else:
        print(f"Missing word: {w1} or {w2}")
```

---

# Step 6: Find Nearest Words

```python
def nearest_words(target_word, top_n=10):
    if target_word not in glove:
        return []

    target_vec = get_vector(target_word)

    scores = []

    for word in glove:
        if word == target_word:
            continue

        sim = cosine_similarity(target_vec, get_vector(word))

        scores.append((word, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_n]
```

```python
target = "king"

neighbors = nearest_words(target, top_n=10)

print("Nearest words to:", target)

for word, score in neighbors:
    print(f"{word:15s} {score:.4f}")
```

---

# Step 7: Compare Multiple Professions

```python
profession_words = [
    "doctor",
    "nurse",
    "engineer",
    "teacher"
]
```

```python
for base_word in profession_words:
    print(f"\nNearest to {base_word}:")
    
    neighbors = nearest_words(base_word, top_n=5)

    for word, score in neighbors:
        print(f"{word:15s} {score:.4f}")
```

---

# Step 8: Word Arithmetic

We test:

$king - man + woman \approx queen$

```python
result_vector = (
    get_vector("king")
    - get_vector("man")
    + get_vector("woman")
)
```

```python
scores = []

for word in glove:
    sim = cosine_similarity(result_vector, get_vector(word))

    scores.append((word, sim))

scores.sort(key=lambda x: x[1], reverse=True)
```

```python
print("Top 10 results for king - man + woman:\n")

for word, score in scores[:10]:
    print(f"{word:15s} {score:.4f}")
```

---

# Step 9: More Arithmetic Experiments

```python
experiments = [
    ("queen", "woman", "man"),
    ("doctor", "man", "woman"),
    ("teacher", "man", "woman")
]
```

```python
for a, b, c in experiments:
    print(f"\nTesting: {a} - {b} + {c}")

    vec = get_vector(a) - get_vector(b) + get_vector(c)

    scores = []

    for word in glove:
        sim = cosine_similarity(vec, get_vector(word))

        scores.append((word, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    for word, score in scores[:5]:
        print(f"{word:15s} {score:.4f}")
```

---

# Step 10: Average Vector of a Small Sentence

We combine word vectors by averaging.

```python
def sentence_vector(sentence):
    words = sentence.lower().split()

    vectors = []

    for word in words:
        if word in glove:
            vectors.append(get_vector(word))

    if len(vectors) == 0:
        return np.zeros(50)

    return np.mean(vectors, axis=0)
```

```python
sentence = "king queen man woman"

vec = sentence_vector(sentence)

print("Sentence:", sentence)

print("Vector shape:", vec.shape)

print("First 10 values:")
print(vec[:10])
```

---

# Step 11: Sentence Similarity

```python
sentences = [
    "king queen",
    "man woman",
    "doctor nurse",
    "banana orange"
]
```

```python
base_sentence = "king man"

base_vec = sentence_vector(base_sentence)

print("Base sentence:", base_sentence)

for s in sentences:
    vec = sentence_vector(s)

    sim = cosine_similarity(base_vec, vec)

    print(f"{s:15s} -> {sim:.4f}")
```

---

# Step 12: Out-of-Vocabulary Check

```python
test_words = [
    "king",
    "dragon",
    "teacher",
    "spaceship"
]
```

```python
for word in test_words:
    if word in glove:
        print(f"{word:10s} -> Found")
    else:
        print(f"{word:10s} -> Missing")
```

---

# Step 13: Build a Mini Similarity Table

```python
words = ["king", "queen", "man", "woman"]
```

```python
print("Cosine Similarity Table:\n")

for w1 in words:
    row = []

    for w2 in words:
        sim = cosine_similarity(
            get_vector(w1),
            get_vector(w2)
        )

        row.append(f"{sim:.3f}")

    print(w1.ljust(8), row)
```

---

# Step 14: Final Exploration

```python
custom_words = [
    "science",
    "technology",
    "teacher",
    "student"
]
```

```python
for word in custom_words:
    if word in glove:
        print(f"\nTop neighbors for {word}:")
        
        for neighbor, score in nearest_words(word, top_n=5):
            print(f"{neighbor:15s} {score:.4f}")
    else:
        print(f"\n{word} not found in vocabulary.")
```
