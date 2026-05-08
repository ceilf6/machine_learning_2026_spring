# Task 9: Logistic Regression for Sentiment Classification

## Objective

We build a simple sentiment classifier.

We use:

* TF-IDF for text features
* Logistic Regression for classification

We predict:

* 1 = Positive
* 0 = Negative

---

# Step 1: Import Libraries

```python
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

---

# Step 2: Load Dataset

```python
DATA_DIR = Path.cwd() / "datasets"

df = pd.read_csv(DATA_DIR / "imdb_balanced_10k.csv")
```

```python
print("Total reviews:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst review preview:")
print(df["text"].iloc[0][:300])

print("\nFirst label:", df["label"].iloc[0])
```

---

# Step 3: Split Train and Test Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
```

```python
print("Testing samples:", len(X_test))

print("\nFirst training review:")
print(X_train.iloc[0][:300])
```

---

# Step 4: Convert Text into TF-IDF Features

```python
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)
```

```python
print("Training shape:", X_train_tfidf.shape)

print("Testing shape:", X_test_tfidf.shape)

print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
```

---

# Step 5: Inspect Vocabulary

```python
feature_names = vectorizer.get_feature_names_out()

print("First 20 vocabulary words:")
print(feature_names[:20])
```

---

# Step 6: Train Logistic Regression

```python
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)
```

---

# Step 7: Make Predictions

```python
y_pred = model.predict(X_test_tfidf)
```

```python
print("First 20 predictions:")
print(y_pred[:20])

print("\nFirst 20 true labels:")
print(y_test.iloc[:20].values)
```

---

# Step 8: Evaluate Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)
```

```python
print("Classification Accuracy:", round(accuracy, 4))
```

---

# Step 9: Print Full Classification Report

```python
print("Classification Report:\n")

print(classification_report(y_test, y_pred))
```

---

# Step 10: Compare Real Reviews with Predictions

```python
for i in range(5):
    print(f"\nReview {i+1}")
    print("-" * 60)

    print(X_test.iloc[i][:400])

    print("\nTrue Label:", y_test.iloc[i])

    print("Predicted Label:", y_pred[i])

    if y_test.iloc[i] == y_pred[i]:
        print("Result: Correct")
    else:
        print("Result: Incorrect")
```

---

# Step 11: Check Important Words

We inspect which words have strong positive or negative weights.

```python
coefficients = model.coef_[0]

top_positive_idx = np.argsort(coefficients)[-10:]

top_negative_idx = np.argsort(coefficients)[:10]
```

```python
print("Top Positive Words:")

for idx in reversed(top_positive_idx):
    print(feature_names[idx], round(coefficients[idx], 4))
```

```python
print("Top Negative Words:")

for idx in top_negative_idx:
    print(feature_names[idx], round(coefficients[idx], 4))
```

---

# Step 12: Predict New Reviews

```python
sample_reviews = [
    "This movie was amazing and unforgettable",
    "This was a terrible boring film",
    "The movie was okay but too long"
]
```

```python
sample_tfidf = vectorizer.transform(sample_reviews)

sample_preds = model.predict(sample_tfidf)
```

```python
for review, pred in zip(sample_reviews, sample_preds):
    print("\nReview:")
    print(review)

    print(
        "Predicted Sentiment:",
        "Positive" if pred == 1 else "Negative"
    )
```
