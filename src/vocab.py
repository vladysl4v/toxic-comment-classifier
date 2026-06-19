from collections import Counter
import json
import string


def tokenize(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


class Vocabulary:
    def __init__(self, max_size=55000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = Counter()

    def build(self, texts):
        for text in texts:
            tokens = tokenize(text)
            self.word_counts.update(tokens)

        for word, count in self.word_counts.most_common(self.max_size):
            if count < self.min_freq:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")

    def save(self, path="data/vocab.json"):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)
        print(f"Vocabulary saved to {path}")

    def load(self, path="data/vocab.json"):
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Vocabulary loaded from {path} — size: {len(self.word2idx)}")

    def encode(self, text):
        tokens = tokenize(text)
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def __len__(self):
        return len(self.word2idx)