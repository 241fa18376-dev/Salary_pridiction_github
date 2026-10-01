"""
Salary Prediction - Preprocessing
---------------------------------
Loads the raw salary dataset, validates/cleans it, performs a reproducible
train/test split, scales numerical features, and saves model-ready arrays.
The dataset contains only numerical features, so categorical encoding is not
required; ohe.pkl is saved as a small metadata object for structural parity
with the classification project.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "salary.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = ["YearsExperience"]
TARGET_COLUMN = "Salary"


def load_and_clean_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the salary dataset and perform basic validation/cleaning."""
    df = pd.read_csv(path)

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    # Basic domain validation for this dataset.
    if (df["YearsExperience"] < 0).any():
        raise ValueError("YearsExperience contains negative values.")

    if len(df) < 5:
        raise ValueError("Not enough valid rows after cleaning.")

    df.attrs["rows_removed_during_cleaning"] = int(removed)
    return df


def preprocess() -> dict:
    """Run preprocessing and save all processed artifacts."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train)
    X_test_final = scaler.transform(X_test)

    np.save(PROCESSED_DIR / "X_train_final.npy", X_train_final)
    np.save(PROCESSED_DIR / "X_test_final.npy", X_test_final)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # No categorical columns exist in the supplied dataset.
    joblib.dump(
        {
            "used": False,
            "encoder": None,
            "categorical_features": [],
            "reason": "Salary dataset contains only numerical feature(s).",
        },
        MODELS_DIR / "ohe.pkl",
    )

    metadata = {
        "project": "Salary Prediction",
        "dataset_file": "data/raw/salary.csv",
        "source_shape": [int(raw_shape[0]), int(raw_shape[1])] if (raw_shape := pd.read_csv(RAW_PATH).shape) else None,
        "cleaned_shape": [int(df.shape[0]), int(df.shape[1])],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "task": "regression",
        "test_size": 0.20,
        "random_state": 42,
        "train_shape": [int(X_train_final.shape[0]), int(X_train_final.shape[1])],
        "test_shape": [int(X_test_final.shape[0]), int(X_test_final.shape[1])],
        "scaling": "StandardScaler fitted on training data only",
        "categorical_encoding": "Not used; no categorical features in source dataset",
        "rows_removed_during_cleaning": int(df.attrs.get("rows_removed_during_cleaning", 0)),
        "feature_count_after_preprocessing": int(X_train_final.shape[1]),
        "dataset_summary": {
            "years_experience_min": float(df["YearsExperience"].min()),
            "years_experience_max": float(df["YearsExperience"].max()),
            "salary_min": float(df["Salary"].min()),
            "salary_max": float(df["Salary"].max()),
            "salary_mean": float(df["Salary"].mean()),
            "years_experience_salary_correlation": float(
                df["YearsExperience"].corr(df["Salary"])
            ),
        },
    }

    with open(PROCESSED_DIR / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    metadata = preprocess()
    print("Preprocessing completed.")
    print(json.dumps(metadata, indent=2))
