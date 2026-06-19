import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ToxicDataset(Dataset):
    """
    A single dataset class that works for baseline, LSTM, and BERT/GPT.

    mode="raw"   -> returns (text_string, label_tensor)
                    use this for the keyword baseline or any custom preprocessing

    mode="lstm"  -> returns (token_id_tensor, label_tensor)
                    needs a Vocabulary object passed in

    mode="bert"  -> returns (input_ids, attention_mask, label_tensor)
                    needs a HuggingFace tokenizer passed in
    """

    def __init__(self, csv_path, id_path, mode="raw", vocab=None, tokenizer=None, max_len=256):
        """
        csv_path  : path to the full train.csv
        id_path   : path to train_ids.csv or val_ids.csv (produced by split.py)
        mode      : "raw" | "lstm" | "bert"
        vocab     : Vocabulary instance (only needed for mode="lstm")
        tokenizer : HuggingFace tokenizer (only needed for mode="bert")
        max_len   : max sequence length for lstm / bert
        """
        full_df = pd.read_csv(csv_path)
        ids = pd.read_csv(id_path)["id"]
        self.df = full_df[full_df["id"].isin(ids)].reset_index(drop=True)

        self.mode = mode
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["comment_text"]
        # labels as a float tensor — shape (6,)
        labels = torch.tensor(row[LABELS].values.astype(float), dtype=torch.float32)

        if self.mode == "raw":
            # just hand back the raw string; the baseline predictor handles the rest
            return text, labels

        elif self.mode == "lstm":
            # encode with our custom Vocabulary, then truncate
            token_ids = self.vocab.encode(text)[:self.max_len]
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            return token_tensor, labels

        elif self.mode == "bert":
            # Tokenize WITHOUT padding — we'll do dynamic padding in collate_fn
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)        # variable length
            attention_mask = encoding["attention_mask"].squeeze(0)  # variable length
            return input_ids, attention_mask, labels

        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Choose: raw | lstm | bert")


# ── Collate functions ──────────────────────────────────────────────────────────
# A collate_fn tells DataLoader how to stack samples into a batch.
# Both lstm and bert use dynamic padding — pad to the longest sequence in
# each batch, not a fixed global max_len. raw mode needs no collate_fn
# since it just returns strings.

def collate_lstm(batch):
    """Pads variable-length LSTM sequences to the longest in the batch."""
    sequences, labels = zip(*batch)
    # pad_sequence stacks tensors and pads the shorter ones with 0 (<PAD> index)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, labels


def collate_bert(batch):
    """Same idea as collate_lstm — dynamic padding to longest in the batch.
    BERT returns two tensors per sample (input_ids + attention_mask), so we pad both."""
    input_ids, attention_masks, labels = zip(*batch)
    # pad input_ids with tokenizer's pad id (0 for most BERT models)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    # pad attention_mask with 0 — meaning "ignore this position"
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return input_ids, attention_masks, labels


# ── Helper to build a DataLoader in one call ───────────────────────────────────

def get_dataloader(csv_path, id_path, mode, batch_size=32,
                   shuffle=True, vocab=None, tokenizer=None, max_len=256):
    """
    Convenience wrapper — returns a ready-to-use DataLoader.

    Example usage:
        # Baseline
        loader = get_dataloader("data/train.csv", "data/train_ids.csv", mode="raw")

        # LSTM
        loader = get_dataloader("data/train.csv", "data/train_ids.csv",
                                mode="lstm", vocab=my_vocab, max_len=200)

        # BERT
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        loader = get_dataloader("data/train.csv", "data/train_ids.csv",
                                mode="bert", tokenizer=tok, max_len=128)
    """
    dataset = ToxicDataset(csv_path, id_path, mode=mode,
                           vocab=vocab, tokenizer=tokenizer, max_len=max_len)

    collate_fn = {"lstm": collate_lstm, "bert": collate_bert}.get(mode, None)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return loader


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    CSV = "data/train.csv"
    TRAIN_IDS = "data/train_ids.csv"

    # --- raw mode ---
    print("=== raw mode ===")
    loader = get_dataloader(CSV, TRAIN_IDS, mode="raw", batch_size=4)
    texts, labels = next(iter(loader))
    print(f"texts:  {type(texts)}, len={len(texts)}")
    print(f"labels: {labels.shape}")   # (4, 6)

    # --- lstm mode ---
    print("\n=== lstm mode ===")
    from vocab import Vocabulary
    vocab = Vocabulary()
    vocab.load("data/vocab.json")   # build this first with vocab.py
    loader = get_dataloader(CSV, TRAIN_IDS, mode="lstm", batch_size=4, vocab=vocab)
    tokens, labels = next(iter(loader))
    print(f"tokens: {tokens.shape}")   # (4, padded_len)
    print(f"labels: {labels.shape}")   # (4, 6)

    # --- bert mode ---
    print("\n=== bert mode ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader = get_dataloader(CSV, TRAIN_IDS, mode="bert", batch_size=4, tokenizer=tok, max_len=128)
    input_ids, attn_mask, labels = next(iter(loader))
    print(f"input_ids:    {input_ids.shape}")    # (4, 128)
    print(f"attn_mask:    {attn_mask.shape}")    # (4, 128)
    print(f"labels:       {labels.shape}")       # (4, 6)