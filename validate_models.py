
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).parent
DATA = ROOT / "data"

RANDOM_STATE = 42
N_SPLITS = 5


# ==========================================
# MODEL PIPELINE
# Same configuration as train_models.py
# ==========================================

def create_pipeline(numeric, categorical=None):

    categorical = categorical or []

    preprocessor = ColumnTransformer([

        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            numeric
        ),

        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categorical
        )

    ])

    model = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])


# ==========================================
# CROSS VALIDATION EVALUATION
# ==========================================

def evaluate_cv(
    name,
    X,
    y,
    numeric,
    categorical=None,
    groups=None
):

    print("\n" + "=" * 65)
    print(f"5-FOLD CROSS VALIDATION: {name.upper()}")
    print("=" * 65)

    categorical = categorical or []

    if groups is not None:

        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        splits = list(
            cv.split(X, y, groups=groups)
        )

        print("Validation: Subject-wise grouped splitting")

    else:

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        splits = list(cv.split(X, y))

        print("Validation: Stratified splitting")

    # Out-of-fold predictions
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.full(len(y), np.nan)

    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        splits, start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Verify no subject leakage
        if groups is not None:

            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])

            assert train_groups.isdisjoint(test_groups), (
                "Subject leakage detected!"
            )

            print(
                f"\nFold {fold}: "
                f"{len(train_groups)} training subjects, "
                f"{len(test_groups)} testing subjects"
            )

        else:

            print(f"\nFold {fold}")

        print(
            f"Training records: {len(train_idx)} | "
            f"Testing records: {len(test_idx)}"
        )

        # Fresh pipeline for every fold
        pipeline = create_pipeline(
            numeric,
            categorical
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)

        positive_index = list(
            pipeline.classes_
        ).index(1)

        positive_probabilities = probabilities[
            :, positive_index
        ]

        oof_pred[test_idx] = predictions
        oof_proba[test_idx] = positive_probabilities

        fold_acc = accuracy_score(
            y_test, predictions
        )

        fold_balanced_acc = balanced_accuracy_score(
            y_test, predictions
        )

        fold_f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        fold_scores.append({
            "accuracy": fold_acc,
            "balanced_accuracy": fold_balanced_acc,
            "f1": fold_f1
        })

        print(f"Accuracy: {fold_acc:.4f}")
        print(
            f"Balanced Accuracy: "
            f"{fold_balanced_acc:.4f}"
        )
        print(f"F1 Score: {fold_f1:.4f}")

    # ======================================
    # OVERALL OUT-OF-FOLD RESULTS
    # ======================================

    assert np.all(oof_pred != -1)
    assert np.all(np.isfinite(oof_proba))

    cm = confusion_matrix(
        y, oof_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 65)
    print("OVERALL OUT-OF-FOLD RESULTS")
    print("-" * 65)

    print(
        "\nActual class distribution:\n",
        y.value_counts().sort_index()
    )

    print(
        "\nPredicted class distribution:\n",
        pd.Series(oof_pred).value_counts().sort_index()
    )

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nTN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            oof_pred,
            labels=[0, 1],
            target_names=[
                "No Disease (0)",
                "Disease (1)"
            ],
            zero_division=0
        )
    )

    print("\nOverall Metrics:")

    metrics = {
        "accuracy": accuracy_score(y, oof_pred),

        "balanced_accuracy": balanced_accuracy_score(
            y, oof_pred
        ),

        "precision": precision_score(
            y, oof_pred, zero_division=0
        ),

        "recall": recall_score(
            y, oof_pred, zero_division=0
        ),

        "f1": f1_score(
            y, oof_pred, zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y, oof_proba
        )
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nMean Fold Metrics:")

    fold_df = pd.DataFrame(fold_scores)

    print(fold_df.mean().round(4))

    print("=" * 65)

    return metrics


# ==========================================
# 1. KIDNEY DISEASE
# ==========================================

def validate_kidney():

    print("\nLoading Kidney Dataset...")

    k = pd.read_csv(
        DATA / "kidney.csv"
    ).drop_duplicates().reset_index(drop=True)

    k.columns = [
        str(c).strip().lower()
        for c in k.columns
    ]

    for c in k.select_dtypes(include="object"):

        k[c] = (
            k[c]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "?": np.nan,
                "\\?": np.nan,
                "": np.nan,
                "nan": np.nan
            })
        )

    target = next(
        c for c in k.columns
        if c in ["classification", "class"]
    )

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

    for c in numeric:

        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        )

    valid = y.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    # Check for duplicate feature rows after cleaning
    feature_keys = pd.util.hash_pandas_object(
        X.astype("string").fillna("<NA>"),
        index=False
    )

    duplicate_mask = feature_keys.duplicated(
        keep=False
    )

    duplicate_labels = (
        pd.DataFrame({
            "feature_key": feature_keys,
            "target": y
        })
        .groupby("feature_key")["target"]
        .nunique()
    )

    conflicting = int((duplicate_labels > 1).sum())

    print("\nKidney dataset shape:", X.shape)
    print(
        "Repeated feature rows:",
        int(duplicate_mask.sum())
    )
    print(
        "Feature groups with conflicting labels:",
        conflicting
    )

    if conflicting:
        raise ValueError(
            "Identical feature rows have conflicting labels. "
            "Investigate before validation."
        )

    # Keep identical feature rows in the same fold
    groups = feature_keys.astype(str)

    return evaluate_cv(
        name="kidney",
        X=X,
        y=y,
        numeric=numeric,
        categorical=categorical,
        groups=groups
    )


# ==========================================
# 2. PARKINSON'S DISEASE
# ==========================================

def validate_parkinsons():

    print("\nLoading Parkinson's Dataset...")

    p = pd.read_csv(
        DATA / "parkinsons.csv"
    )

    # Extract subject IDs
    groups = (
        p["name"]
        .astype(str)
        .str.extract(r"(S\d+)")[0]
    )

    if groups.isna().any():
        raise ValueError(
            "Some Parkinson's subject IDs could not be extracted."
        )

    X = p.drop(
        columns=["name", "status"]
    )

    y = p["status"].astype(int)

    # Check subject-level labels
    subject_labels = (
        pd.DataFrame({
            "subject": groups,
            "status": y
        })
        .groupby("subject")["status"]
        .nunique()
    )

    if (subject_labels > 1).any():
        raise ValueError(
            "Some subjects have inconsistent labels."
        )

    print("\nTotal recordings:", len(p))
    print("Unique subjects:", groups.nunique())

    print("\nSubject-level class distribution:")

    print(
        p.assign(subject=groups)
        .groupby("subject")["status"]
        .first()
        .value_counts()
    )

    numeric = list(X.columns)

    return evaluate_cv(
        name="parkinsons",
        X=X,
        y=y,
        numeric=numeric,
        groups=groups
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    kidney_results = validate_kidney()

    parkinsons_results = validate_parkinsons()

    print("\n\nFINAL CROSS-VALIDATION SUMMARY")

    print("\nKidney:", kidney_results)
    print("\nParkinson's:", parkinsons_results)

    print("\nValidation completed successfully.")
