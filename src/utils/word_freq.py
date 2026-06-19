import csv
import os
import string

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tokenize(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def plot_word_freq(word_counts, save_path, max_freq=None):
    total_unique = len(word_counts)

    fig, ax = plt.subplots(figsize=(12, 6))

    if max_freq is not None:
        freq_buckets = Counter(c for c in word_counts.values() if 1 <= c <= max_freq)
        x = list(range(1, max_freq + 1))
        y = [freq_buckets.get(i, 0) for i in x]
        ax.bar(x, y, width=0.6)
        ax.set_xticks(x)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{int(val):,}"))
        ax.set_title(f"Word Frequency Distribution (1–{max_freq} appearances)\nTotal unique words: {total_unique:,}")
    else:
        freqs = list(word_counts.values())
        ax.hist(freqs, bins=100, log=True)
        ax.set_title(f"Word Frequency Distribution (full)\nTotal unique words: {total_unique:,}")

    ax.set_xlabel("Number of appearances")
    ax.set_ylabel("Number of words")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


def save_freq_examples(word_counts, save_path, n_examples=15):
    words_by_freq = {}
    for word, count in word_counts.items():
        if 1 <= count <= 10:
            words_by_freq.setdefault(count, []).append(word)

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["frequency", "total_words"] + [f"example_{i+1}" for i in range(n_examples)]
        writer.writerow(header)
        for freq in range(1, 11):
            examples = words_by_freq.get(freq, [])[:n_examples]
            examples += [""] * (n_examples - len(examples))
            writer.writerow([freq, len(words_by_freq.get(freq, []))] + examples)

    print(f"Saved to {save_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data/analysis"), exist_ok=True)

    train_df = pd.read_csv(os.path.join(BASE_DIR, "data/train.csv"))
    texts = train_df["comment_text"].values

    word_counts = Counter()
    for text in texts:
        tokens = tokenize(text)
        word_counts.update(tokens)

    plot_word_freq(word_counts, os.path.join(BASE_DIR, "graphs/word_freq_zoom.png"), max_freq=10)
    plot_word_freq(word_counts, os.path.join(BASE_DIR, "graphs/word_freq_full.png"))
    save_freq_examples(word_counts, os.path.join(BASE_DIR, "data/analysis/word_freq_examples.csv"))