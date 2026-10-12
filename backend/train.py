"""
train.py
--------
Trains the three deficiency-risk models FROM THE ACTUAL NOTEBOOK
(ICD_Project.ipynb, in this same folder) and saves the artifacts the
FastAPI backend loads at inference time.

The notebook is the single source of truth for: dataset generation, the
deficiency target rules, and data cleaning. Edit any of that in the .ipynb
and re-run this script (or just save the notebook - see main.py's
background watcher) to retrain on the new logic automatically.

On top of whatever dataframe the notebook produces, this script applies a
fixed, robust modeling path so results stay reliable regardless of how the
notebook is edited:
  1. Per-column LabelEncoder for each categorical feature (persisted).
  2. One StandardScaler fit across the full feature matrix (persisted).
  3. A RandomForestClassifier(class_weight="balanced") per target - chosen
     over a plain DecisionTree because its averaged votes across trees
     produce a graded predict_proba, which the frontend needs to show
     Low / Moderate / High risk bands instead of only 0% or 100%.

Usage:
    python train.py                       # uses ICD_Project.ipynb in this folder
    python train.py --notebook other.ipynb
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from notebook_runtime import extract_clean_dataframe, TARGET_COLUMNS, CATEGORICAL_COLUMNS

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_NOTEBOOK = BASE_DIR / "ICD_Project.ipynb"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Belt-and-braces re-clean in case the notebook snapshot still has NaNs
    (e.g. it was captured right after target creation, before its own
    cleaning cell ran)."""
    df = df.copy()
    for column in df.columns:
        if df[column].isna().any():
            if df[column].dtype in ["int64", "float64"]:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])
    return df.drop_duplicates().reset_index(drop=True)


def infer_input_features(df: pd.DataFrame) -> list[str]:
    """Every column except the three targets, in the notebook's own column
    order - so if the notebook's feature-engineering cells add or remove a
    column, the served model picks it up automatically."""
    return [c for c in df.columns if c not in TARGET_COLUMNS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()

    print(f"Executing notebook: {args.notebook}")
    df = extract_clean_dataframe(args.notebook)
    df = clean(df)
    print(f"Extracted dataframe: {df.shape[0]} rows, columns={list(df.columns)}")

    input_features = infer_input_features(df)
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    print("Encoding categorical columns:", categorical_cols)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.joblib")

    X = df[input_features]

    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    metrics = {}
    for target in TARGET_COLUMNS:
        if target not in df.columns:
            print(f"WARNING: target '{target}' not found in notebook dataframe, skipping.")
            continue

        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        candidates = {
            "LogisticRegression": LogisticRegression(
                random_state=RANDOM_STATE, solver="lbfgs", class_weight="balanced", max_iter=1000
            ),
            "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", max_depth=8
            ),
        }

        target_metrics = {}
        for name, clf in candidates.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            target_metrics[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            }
            print(f"{target:30s} {name:18s} f1={target_metrics[name]['f1']}")

        deployed = candidates["RandomForest"]
        joblib.dump(deployed, MODELS_DIR / f"{target}_model.joblib")

        metrics[target] = {"candidates": target_metrics, "deployed_model": "RandomForest"}

    with open(MODELS_DIR / "feature_order.json", "w") as f:
        json.dump(input_features, f, indent=2)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts to", MODELS_DIR)


if __name__ == "__main__":
    main()
