# Hugging Face Ready-to-Use Sentiment Analysis


## Objective

We use Hugging Face pretrained pipelines.

We do not train.

We directly use a ready-made sentiment model.

We use China mirror.

We use PyTorch backend.

We test on `imdb_top_500.csv`.

* 1 = Positive
* 0 = Negative

---

## Step 1: Install Libraries

```python
!pip install transformers pandas torch -q
```

---

## Step 2: Set Hugging Face Mirror

```python
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

## Step 3: Import Libraries

```python
import pandas as pd

from transformers import pipeline
```

---

## Step 4: Load Dataset

```python
df = pd.read_csv("./datasets/imdb_top_500.csv")
```

```python
print("Dataset size:", len(df))

print("Columns:", df.columns.tolist())

print("\nFirst review preview:")
print(df["text"].iloc[0][:300])

print("\nFirst label:", df["label"].iloc[0])

print("First rating:", df["rating"].iloc[0])
```

---

## Step 5: Build Hugging Face Pipeline

We use a stable pretrained model.

```python
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)
```

```python
print("Pipeline loaded successfully.")

print("Model name:", classifier.model.name_or_path)
```

---

## Step 6: Check Label Style

```python
print("Model labels:")
print(classifier.model.config.id2label)
```

---

## Step 7: Test One Positive Sentence

```python
result = classifier("This movie was fantastic and exciting")
```

```python
print("Prediction:")
print(result)
```

---

## Step 8: Test One Negative Sentence

```python
result = classifier("This movie was terrible and boring")
```

```python
print("Prediction:")
print(result)
```

---

## Step 9: Predict First 10 IMDB Reviews

```python
sample_reviews = df["text"].iloc[:10].tolist()
```

```python
predictions = classifier(
    sample_reviews,
    truncation=True
)
```

```python
print("Total predictions:", len(predictions))

print("\nFirst prediction:")
print(predictions[0])
```

---

## Step 10: Convert POSITIVE and NEGATIVE into 1 and 0

```python
def hf_to_label(pred):
    return 1 if pred["label"] == "POSITIVE" else 0
```

```python
converted_preds = [
    hf_to_label(pred)
    for pred in predictions
]
```

```python
print("Converted predictions:")
print(converted_preds)
```

---

## Step 11: Compare with Real Labels

```python
true_labels = df["label"].iloc[:10].tolist()
```

```python
for i in range(10):
    print(f"\nReview {i+1}")
    print("-" * 50)

    print(df["text"].iloc[i][:200])

    print("\nTrue Label:", true_labels[i])

    print("Predicted Label:", converted_preds[i])

    if true_labels[i] == converted_preds[i]:
        print("Result: Correct")
    else:
        print("Result: Incorrect")
```

---

## Step 12: Run on First 50 Reviews

```python
subset_size = 50

reviews = df["text"].iloc[:subset_size].tolist()

true_labels = df["label"].iloc[:subset_size].tolist()
```

```python
predictions = classifier(
    reviews,
    truncation=True,
    batch_size=8
)
```

```python
pred_labels = [
    hf_to_label(pred)
    for pred in predictions
]
```

```python
print("Finished predicting", len(pred_labels), "reviews.")
```

---

## Step 13: Compute Accuracy

```python
correct = sum(
    p == y
    for p, y in zip(pred_labels, true_labels)
)

accuracy = correct / len(true_labels)
```

```python
print("Accuracy on first", subset_size, "reviews:", accuracy)
```

---

## Step 14: View Confidence Scores

```python
for i in range(5):
    print(f"\nReview {i+1}")

    print("HF Label:", predictions[i]["label"])

    print("Confidence Score:", predictions[i]["score"])
```

---

## Step 15: Compare Real Reviews More Clearly

```python
for i in range(5):
    print(f"\nReview {i+1}")
    print("-" * 60)

    print(reviews[i][:300])

    print("\nTrue Label:", true_labels[i])

    print("Predicted Label:", pred_labels[i])

    if true_labels[i] == pred_labels[i]:
        print("Result: Correct")
    else:
        print("Result: Incorrect")
```

---

## Step 16: Predict Custom Reviews

```python
custom_reviews = [
    "This movie was amazing with brilliant acting",
    "I hated this film so much",
    "It was okay, not bad, not great"
]
```

```python
custom_preds = classifier(
    custom_reviews,
    truncation=True
)
```

```python
for review, pred in zip(custom_reviews, custom_preds):
    print("\nReview:")
    print(review)

    print("HF Label:", pred["label"])

    print("Numeric Label:", hf_to_label(pred))

    print("Score:", pred["score"])
```

---

## Step 17: Simple Batch Prediction Function

```python
def predict_sentiment(text_list):
    preds = classifier(
        text_list,
        truncation=True
    )

    return [
        {
            "text": text,
            "hf_label": pred["label"],
            "numeric_label": hf_to_label(pred),
            "score": pred["score"]
        }
        for text, pred in zip(text_list, preds)
    ]
```

```python
sample_results = predict_sentiment([
    "What a wonderful movie",
    "This was a disaster"
])

for item in sample_results:
    print("\nText:", item["text"])

    print("HF Label:", item["hf_label"])

    print("Numeric Label:", item["numeric_label"])

    print("Score:", item["score"])
```


## Your task:

As we can see for the huggingface sentiment analysis task,

with: 

```python
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)
```

the model is not strong enough (88% of accuracy for the first 50 reviews, just in per with Bag Of Word + Logistic Regression). the model is kind of small and out-dated.

So, i would like you guys to check out other models.

SEND YOUR FINDINGS HERE IN THE WECHAT GROUP, the model and the accuracy.

Ideally, we should be able to find a good model with, let's say, higher than 95% of accuracy.