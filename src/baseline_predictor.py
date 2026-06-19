import re
from collections import Counter

import nltk
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
import numpy as np

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    return [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]


def top_keywords(df, label, n=15):
    mask = df[label] == 1
    words = []
    for text in df.loc[mask, "comment_text"]:
        words.extend(clean(text))
    return {word for word, _ in Counter(words).most_common(n)}


def predict(text, keyword_sets):
    words = set(clean(text))
    return [1 if words & kws else 0 for kws in keyword_sets]


def main():
    df = pd.read_csv("data/train.csv")

    train = df.sample(frac=0.8, random_state=42)
    val = df.drop(train.index).reset_index(drop=True)
    print(f"Train: {len(train)}  Val: {len(val)}\n")

    keyword_sets = [top_keywords(train, label) for label in LABELS]

    for label, kws in zip(LABELS, keyword_sets):
        print(f"{label}: {sorted(kws)}")
    print()

    preds = val["comment_text"].apply(lambda t: predict(t, keyword_sets))
    pred_df = pd.DataFrame(preds.tolist(), columns=LABELS)

    y_true = val[LABELS].values
    y_pred = pred_df.values

    # per label metrics
    print(f"{'Label':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 72)
    for i, label in enumerate(LABELS):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        print(f"{label:<20} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {auc:>10.3f}")

    # overall metrics
    print("\n=== Overall Metrics ===\n")
    print(f"Overall Accuracy:   {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro F1:           {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    print(f"Macro ROC-AUC:      {roc_auc_score(y_true, y_pred, average='macro'):.3f}")


if __name__ == "__main__":
    main()