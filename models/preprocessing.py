import pandas as pd

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

df = pd.read_csv("data/raw/train.csv")

# Basic checks
print(df.shape)
print(df.columns)
print(df.head())

# Keep only useful columns
df = df[["comment_text"] + LABELS]

# Handle missing text
df["comment_text"] = df["comment_text"].fillna("")

# Remove exact duplicates if any
df = df.drop_duplicates()

# Save clean version
df.to_csv("data/processed/train_clean.csv", index=False)
df["comment_text"] = df["comment_text"].str.replace("\n", " ", regex=False)
df["comment_text"] = df["comment_text"].str.replace("\t", " ", regex=False)
df["comment_text"] = df["comment_text"].str.strip()

print(df.shape)
print(df.columns)
print(df.head())
print(df.isnull().sum())
print(df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum())

df["any_toxic"] = df[LABELS].sum(axis=1) > 0

print("Non-toxic comments:", (~df["any_toxic"]).sum())
print("Toxic comments:", df["any_toxic"].sum())

print("\nPercentage non-toxic:")
print((~df["any_toxic"]).mean() * 100)

print("\nPercentage toxic:")
print(df["any_toxic"].mean() * 100)