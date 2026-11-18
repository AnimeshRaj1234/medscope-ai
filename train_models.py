
"""Train four educational disease-classification pipelines."""

from pathlib import Path

import requests
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.datasets import load_breast_cancer


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).parent

MODELS = ROOT / "models"
DATA = ROOT / "data"

MODELS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# DATASET DOWNLOADER
# ==========================================

def fetch(url, path):

    if not path.exists():

        print(f"\nDownloading: {path.name}")

        response = requests.get(
            url,
            timeout=45
        )

        response.raise_for_status()

        path.write_bytes(response.content)

    return path


# ==========================================
# MODEL TRAINING FUNCTION
# ==========================================

def train(
    name,
    X,
    y,
    numeric,
    categorical=None,
    prefer="rf"
):

    categorical = categorical or []
    numeric = list(numeric)

    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    # ---------------- DATA VALIDATION ----------------

    if X.empty:
        raise ValueError(f"{name}: Empty feature dataset.")

    if len(X) != len(y):
        raise ValueError(f"{name}: X and y lengths differ.")

    if y.isna().any():
        raise ValueError(f"{name}: Missing target values.")

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"{name}: Expected binary target labels 0 and 1."
        )

    print(f"\nDataset shape: {X.shape}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # ---------------- PREPROCESSING ----------------

    transformers = []

    if numeric:

        transformers.append(
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())
                ]),
                numeric
            )
        )

    if categorical:

        transformers.append(
            (
                "cat",
                Pipeline([
                    (
                        "impute",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]),
                categorical
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # ---------------- MODEL SELECTION ----------------

    if prefer == "rf":

        model = RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    elif prefer == "extra":

        model = ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    else:

        raise ValueError(f"Unknown model type: {prefer}")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # ---------------- TRAIN TEST SPLIT ----------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ---------------- INITIAL TRAINING ----------------

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)

    positive_index = list(
        pipeline.classes_
    ).index(1)

    positive_probabilities = probabilities[
        :, positive_index
    ]

    # ---------------- EVALUATION ----------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    auc = roc_auc_score(
        y_test,
        positive_probabilities
    )

    metrics = {
        "accuracy": round(
            float(accuracy_score(y_test, predictions)), 4
        ),

        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_test, predictions)), 4
        ),

        "precision": round(
            float(precision_score(
                y_test, predictions, zero_division=0
            )), 4
        ),

        "recall": round(
            float(recall_score(
                y_test, predictions, zero_division=0
            )), 4
        ),

        "f1": round(
            float(f1_score(
                y_test, predictions, zero_division=0
            )), 4
        ),

        "roc_auc": round(float(auc), 4)
    }

    # ---------------- DIAGNOSTICS ----------------

    print("\n" + "=" * 60)
    print(f"MODEL: {name.upper()}")
    print("=" * 60)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nActual class distribution:")
    print(y_test.value_counts().sort_index())

    print("\nPredicted class distribution:")
    print(pd.Series(predictions).value_counts().sort_index())

    print("\nConfusion Matrix:")
    print(cm)

    print("\nConfusion Matrix Interpretation:")
    print(f"True Negatives  (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives  (TP): {tp}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "No Disease (0)",
                "Disease (1)"
            ],
            zero_division=0
        )
    )

    print("\nHoldout Evaluation Metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # ---------------- FINAL TRAINING ----------------
    # Retrain on the complete dataset after evaluation.
    # Holdout metrics above remain unchanged.

    print("\nRetraining final pipeline on complete dataset...")

    pipeline.fit(X, y)

    # ---------------- SAVE MODEL ----------------

    artifact = {
        "pipeline": pipeline,
        "features": list(X.columns),
        "metrics": metrics,
        "classes": [
            str(c) for c in pipeline.classes_
        ],
        "training_samples": len(X),
        "evaluation_samples": len(X_test),
        "evaluation_method": "Stratified 75:25 holdout",
        "model_type": prefer
    }

    output_path = MODELS / f"{name}.joblib"

    joblib.dump(
        artifact,
        output_path
    )

    print("\nFinal model saved:", output_path)
    print("Final training samples:", len(X))
    print("=" * 60)

    return metrics


# ==========================================
# 1. HEART DISEASE
# ==========================================

def train_heart():

    print("\n\n========== HEART DISEASE ==========")

    h = pd.read_csv(
        DATA / "heart.csv"
    )

    h = h.drop_duplicates().reset_index(drop=True)

    print("Heart unique records:", len(h))

    if "target" not in h.columns:
        raise ValueError("Heart dataset missing target column.")

    X = h.drop(columns="target")
    y = pd.to_numeric(h["target"], errors="coerce")

    if y.isna().any():
        raise ValueError("Invalid heart target values.")

    numeric = list(X.columns)

    return train(
        name="heart",
        X=X,
        y=y.astype(int),
        numeric=numeric,
        prefer="rf"
    )


# ==========================================
# 2. DIABETES
# ==========================================

def train_diabetes():

    print("\n\n========== DIABETES ==========")

    url = (
        "https://huggingface.co/datasets/"
        "khoaguin/pima-indians-diabetes-database/"
        "resolve/main/diabetes.csv"
    )

    path = fetch(
        url,
        DATA / "diabetes.csv"
    )

    d = pd.read_csv(path)

    d = d.drop_duplicates().reset_index(drop=True)

    target = (
        "Outcome"
        if "Outcome" in d.columns
        else "y"
    )

    if target not in d.columns:
        raise ValueError("Diabetes target column not found.")

    X = d.drop(columns=target)
    y = pd.to_numeric(d[target], errors="coerce")

    if y.isna().any():
        raise ValueError("Invalid diabetes target values.")

    numeric = list(X.columns)

    return train(
        name="diabetes",
        X=X,
        y=y.astype(int),
        numeric=numeric,
        prefer="rf"
    )


# ==========================================
# 3. CHRONIC KIDNEY DISEASE
# ==========================================

def train_kidney():

    print("\n\n========== KIDNEY DISEASE ==========")

    url = (
        "https://raw.githubusercontent.com/"
        "nimittaswala/Kidney-Disease-Prediction/"
        "main/CKD_Dataset.csv"
    )

    path = fetch(
        url,
        DATA / "kidney.csv"
    )

    k = pd.read_csv(path)

    k = k.drop_duplicates().reset_index(drop=True)

    k.columns = [
        str(c).strip().lower()
        for c in k.columns
    ]

    for column in k.select_dtypes(include="object"):

        k[column] = (
            k[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "?": np.nan,
                "": np.nan,
                "nan": np.nan
            })
        )

    target = next(
        (
            c for c in k.columns
            if c in ["classification", "class"]
        ),
        None
    )

    if target is None:
        raise ValueError("Kidney target column not found.")

    k = k.drop(
        columns=[
            c for c in ["id"]
            if c in k.columns
        ]
    )

    y = (
        k[target]
        .astype(str)
        .str.strip()
        .str.replace("\t", "", regex=False)
        .map({
            "ckd": 1,
            "notckd": 0
        })
    )

    X = k.drop(columns=target)

    numeric_candidates = [
        "age", "bp", "sg", "al", "su",
        "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "wc", "rc"
    ]

    numeric = [
        c for c in numeric_candidates
        if c in X.columns
    ]

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    for column in numeric:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    valid = y.notna()

    X = X.loc[valid].reset_index(drop=True)

    y = (
        y.loc[valid]
        .astype(int)
        .reset_index(drop=True)
    )

    return train(
        name="kidney",
        X=X,
        y=y,
        numeric=numeric,
        categorical=categorical,
        prefer="rf"
    )


# ==========================================
# 4. BREAST CANCER
# ==========================================

def train_breast_cancer():

    print("\n\n========== BREAST CANCER ==========")

    bc = load_breast_cancer(
        as_frame=True
    )

    X = bc.data

    # sklearn label 0 = malignant
    # Our positive label 1 = malignant

    y = pd.Series(
        (bc.target == 0).astype(int),
        index=X.index
    )

    return train(
        name="breast_cancer",
        X=X,
        y=y,
        numeric=list(X.columns),
        prefer="rf"
    )


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    results = {}

    results["heart"] = train_heart()

    results["diabetes"] = train_diabetes()

    results["kidney"] = train_kidney()

    results["breast_cancer"] = train_breast_cancer()

    print("\n\n" + "=" * 60)
    print("FINAL FOUR-DISEASE TRAINING SUMMARY")
    print("=" * 60)

    for name, metrics in results.items():

        print(f"\n{name.upper()}")

        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("\nAll four models trained and saved successfully.")
