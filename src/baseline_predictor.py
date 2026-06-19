import re
from collections import Counter

import nltk
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    return [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]



def predict(text, keyword_sets):
    words = set(clean(text))
    return [1 if words & kws else 0 for kws in keyword_sets]


def main():
    from dataset import get_dataloader

    train_loader = get_dataloader("data/train.csv", "data/train_ids.csv", mode="raw", batch_size=64, shuffle=False)
    val_loader   = get_dataloader("data/train.csv", "data/val_ids.csv",   mode="raw", batch_size=64, shuffle=False)

    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}\n")

    # accumulate all texts and labels from the train loader
    train_texts, train_labels = [], []
    for texts, labels in train_loader:
        train_texts.extend(texts)
        train_labels.append(labels)
    train_labels = torch.cat(train_labels, dim=0)  # (N, 6)

    # build keyword sets from training data
    keyword_sets = []
    for i, label in enumerate(LABELS):
        mask = train_labels[:, i] == 1
        positive_texts = [t for t, m in zip(train_texts, mask) if m]
        words = []
        for text in positive_texts:
            words.extend(clean(text))
        kws = {word for word, _ in Counter(words).most_common(15)}
        keyword_sets.append(kws)

    for label, kws in zip(LABELS, keyword_sets):
        print(f"{label}: {sorted(kws)}")
    print()

    # run predictions on val
    all_preds, all_labels = [], []
    for texts, labels in val_loader:
        for text in texts:
            all_preds.append(predict(text, keyword_sets))
        all_labels.append(labels)

    y_true = torch.cat(all_labels, dim=0).numpy()   # (N, 6)
    y_pred = np.array(all_preds)                     # (N, 6)

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