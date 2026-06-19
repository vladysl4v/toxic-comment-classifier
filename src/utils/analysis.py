import os
import string

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nltk
import pandas as pd
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def tokenize(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def plot_class_imbalance(df, save_path):
    counts = df[LABELS].sum().sort_values(ascending=False)
    clean = len(df[df[LABELS].sum(axis=1) == 0])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(list(counts.index) + ["clean"], list(counts.values) + [clean])
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Class distribution  |  Total samples: {len(df):,}")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


def plot_label_cooccurrence(df, save_path):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)

    for i, l1 in enumerate(LABELS):
        for j, l2 in enumerate(LABELS):
            matrix[i][j] = ((df[l1] == 1) & (df[l2] == 1)).sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_title("Label co-occurrence")
    plt.colorbar(im)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


def plot_text_length_per_label(df, save_path):
    labels_with_clean = ["clean"] + LABELS
    avg_lengths = []

    for label in labels_with_clean:
        if label == "clean":
            mask = df[LABELS].sum(axis=1) == 0
        else:
            mask = df[label] == 1
        avg = df.loc[mask, "comment_text"].apply(lambda t: len(str(t).split())).mean()
        avg_lengths.append(avg)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels_with_clean, avg_lengths)
    ax.set_xlabel("Label")
    ax.set_ylabel("Average text length (words)")
    ax.set_title("Average text length per label")

    for bar, val in zip(bars, avg_lengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


def plot_top_keywords(df, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    stop_words = set(nltk.corpus.stopwords.words("english"))
    all_labels = ["clean"] + LABELS

    for idx, label in enumerate(all_labels):
        if label == "clean":
            mask = df[LABELS].sum(axis=1) == 0
        else:
            mask = df[label] == 1

        word_counts = Counter()
        for text in df.loc[mask, "comment_text"]:
            tokens = tokenize(text)
            tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
            word_counts.update(tokens)

        top = word_counts.most_common(15)
        words = [w for w, _ in top]
        counts = [c for _, c in top]

        axes[idx].barh(words[::-1], counts[::-1])
        axes[idx].set_title(label)
        axes[idx].set_xlabel("Count")

    axes[-1].set_visible(False)

    plt.suptitle("Top 15 keywords per label", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    nltk.download("stopwords", quiet=True)

    df = pd.read_csv(os.path.join(BASE_DIR, "data/train.csv"))

    plot_class_imbalance(df, os.path.join(GRAPHS_DIR, "class_imbalance.png"))
    plot_label_cooccurrence(df, os.path.join(GRAPHS_DIR, "label_cooccurrence.png"))
    plot_text_length_per_label(df, os.path.join(GRAPHS_DIR, "text_length_per_label.png"))
    plot_top_keywords(df, os.path.join(GRAPHS_DIR, "top_keywords_per_label.png"))