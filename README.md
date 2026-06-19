# Toxic Comment Classifier

Multi-label classifier that detects toxic, obscene, threatening, insulting, and identity-based hate in text comments.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Install uv

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Install dependencies

```bash
uv sync
```

## Data setup

This project uses the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) dataset from Kaggle.

Download and place the following files in the `data/` folder:

```
data/
  train.csv
  test.csv
  test_labels.csv
```

## Generate vocab.json

There is no dedicated script for this. Run the following snippet once to build and save the vocabulary from the training data:

```bash
uv run python -c "
import sys, pandas as pd
sys.path.insert(0, 'src')
from vocab import Vocabulary

texts = pd.read_csv('data/train.csv')['comment_text'].tolist()
vocab = Vocabulary()
vocab.build(texts)
vocab.save('data/vocab.json')
"
```

This creates `data/vocab.json`. The vocabulary keeps words that appear at least twice, capped at 55 000 tokens.

## Exploratory analysis

These are one-time scripts that generate charts into the `graphs/` folder.

### Word frequency

`src/utils/word_freq.py` shows how often each word appears across the training set:

- `graphs/word_freq_zoom.png` — bar chart of words appearing 1–10 times
- `graphs/word_freq_full.png` — full histogram of the whole frequency distribution
- `data/analysis/word_freq_examples.csv` — sample words for each frequency bucket (1–10)

Useful for choosing the `min_freq` cutoff when building the vocabulary.

```bash
uv run python src/utils/word_freq.py
```

### Dataset analysis

`src/utils/analysis.py` generates four charts about the training data:

- `graphs/class_imbalance.png` — how many samples exist per label
- `graphs/label_cooccurrence.png` — heatmap of how often labels appear together
- `graphs/text_length_per_label.png` — average comment length per label
- `graphs/top_keywords_per_label.png` — top 15 words for each label (stopwords removed)

```bash
uv run python src/utils/analysis.py
```
