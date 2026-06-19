import pandas as pd

VAL_FRAC = 0.2
SEED = 42


def main():
    df = pd.read_csv("data/train.csv")
    train = df.sample(frac=1 - VAL_FRAC, random_state=SEED)
    val = df.drop(train.index)

    train["id"].to_csv("data/train_ids.csv", index=False)
    val["id"].to_csv("data/val_ids.csv", index=False)

    print(f"Train: {len(train)}  Val: {len(val)}")
    print("Saved data/train_ids.csv and data/val_ids.csv")


if __name__ == "__main__":
    main()