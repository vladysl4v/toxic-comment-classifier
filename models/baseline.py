# ============================================================
# Baseline Model: TF-IDF + One-vs-Rest Logistic Regression
# Toxic Comment Classification
# ============================================================

from pathlib import Path
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_PATH = Path("data/processed/train_clean.csv")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
RESULTS_DIR = REPORT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.pkl"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "baseline_validation_predictions.csv"


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TEXT_COLUMN = "comment_text"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("Loading data...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Basic validation
required_columns = [TEXT_COLUMN] + LABELS
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")

# Stratification helper
df["any_toxic"] = df[LABELS].sum(axis=1).clip(upper=1)

X = df[TEXT_COLUMN]
y = df[LABELS]


# ------------------------------------------------------------
# Train-validation split
# ------------------------------------------------------------

print("\nCreating stratified train-validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["any_toxic"]
)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

print("\nTraining label distribution:")
print(y_train.sum())

print("\nValidation label distribution:")
print(y_val.sum())


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

tfidf_features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode"
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=50_000,
            min_df=2,
            sublinear_tf=True
        )
    )
])


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

baseline_model = Pipeline([
    ("features", tfidf_features),
    ("classifier", OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    ))
])


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\nTraining baseline model...")

baseline_model.fit(X_train, y_train)

print("Training completed.")


# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------

print("\nMaking validation predictions...")

y_val_proba = baseline_model.predict_proba(X_val)
y_val_pred = (y_val_proba >= THRESHOLD).astype(int)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

print("\nClassification report:")
print(classification_report(
    y_val,
    y_val_pred,
    target_names=LABELS,
    zero_division=0
))

metrics = []

for i, label in enumerate(LABELS):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val[label],
        y_val_pred[:, i],
        average="binary",
        zero_division=0
    )

    roc_auc = roc_auc_score(y_val[label], y_val_proba[:, i])

    metrics.append({
        "model": "TF-IDF + Logistic Regression",
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "support": support
    })


metrics_df = pd.DataFrame(metrics)

macro_auc = roc_auc_score(y_val, y_val_proba, average="macro")
micro_auc = roc_auc_score(y_val, y_val_proba, average="micro")

macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
micro_f1 = f1_score(y_val, y_val_pred, average="micro", zero_division=0)

macro_precision = precision_score(y_val, y_val_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_val, y_val_pred, average="macro", zero_division=0)

overall_metrics = pd.DataFrame([
    {
        "model": "TF-IDF + Logistic Regression",
        "label": "macro_average",
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "roc_auc": macro_auc,
        "support": y_val.shape[0]
    },
    {
        "model": "TF-IDF + Logistic Regression",
        "label": "micro_average",
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": micro_f1,
        "roc_auc": micro_auc,
        "support": y_val.shape[0]
    }
])

metrics_df = pd.concat([metrics_df, overall_metrics], ignore_index=True)

print("\nMetrics:")
print(metrics_df)


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

predictions_df = pd.DataFrame({
    "comment_text": X_val.values
})

for i, label in enumerate(LABELS):
    predictions_df[f"{label}_true"] = y_val[label].values
    predictions_df[f"{label}_proba"] = y_val_proba[:, i]
    predictions_df[f"{label}_pred"] = y_val_pred[:, i]

predictions_df.to_csv(PREDICTIONS_PATH, index=False)


# ------------------------------------------------------------
# Save model and metrics
# ------------------------------------------------------------

joblib.dump(baseline_model, MODEL_PATH)
metrics_df.to_csv(METRICS_PATH, index=False)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")
print(f"Validation predictions saved to: {PREDICTIONS_PATH}")