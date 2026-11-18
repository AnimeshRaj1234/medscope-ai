
import pandas as pd

# ---------------- KIDNEY DATASET ----------------

print("\n========== KIDNEY DATASET ==========")

k = pd.read_csv("data/kidney.csv")

print("Shape:", k.shape)
print("Duplicate rows:", k.duplicated().sum())

print("\nClass distribution:")
print(k["classification"].value_counts(dropna=False))

# ---------------- PARKINSONS DATASET ----------------

print("\n========== PARKINSONS DATASET ==========")

p = pd.read_csv("data/parkinsons.csv")

p["subject"] = p["name"].astype(str).str.extract(r"(S\d+)")[0]

print("Total recordings:", len(p))
print("Unique subjects:", p["subject"].nunique())

print("\nRecordings and class distribution per subject:")

print(
    p.groupby("subject")["status"]
    .agg(
        recordings="count",
        status="first",
        unique_labels="nunique"
    )
    .to_string()
)

print("\nOverall class distribution:")
print(p["status"].value_counts())
