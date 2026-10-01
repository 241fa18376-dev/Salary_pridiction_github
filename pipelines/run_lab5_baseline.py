"""Lab 5 - Data validation and reproducible salary prediction pipeline."""
from __future__ import annotations

print("==============================================================================")
print("Executing Lab 5 - Data validation and reproducible salary prediction pipeline")
print("==============================================================================")

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate
from src.preprocess import FEATURE_COLUMNS, RAW_PATH, TARGET_COLUMN, preprocess
from src.train import train_models

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SEED = 42

ARTIFACTS = (
	"data/processed/X_train_final.npy",
	"data/processed/X_test_final.npy",
	"data/processed/y_train.npy",
	"data/processed/y_test.npy",
	"models/scaler.pkl",
	"models/linear_regression.pkl",
	"models/polynomial_regression.pkl",
	"models/svr.pkl",
	"models/random_forest_baseline.pkl",
	"outputs/model_comparison.csv",
	"outputs/predictions.npy",
	"outputs/actual_vs_predicted.csv",
)


def configure_reproducibility() -> None:
	os.environ["PYTHONHASHSEED"] = str(SEED)
	os.environ["OMP_NUM_THREADS"] = "1"
	os.environ["MKL_NUM_THREADS"] = "1"
	random.seed(SEED)
	np.random.seed(SEED)


def validate_raw_data() -> dict[str, int | bool]:
	if not RAW_PATH.exists():
		raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

	data = pd.read_csv(RAW_PATH)
	required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
	missing = sorted(set(required_columns).difference(data.columns))
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	if data.empty:
		raise ValueError("Raw dataset is empty.")

	values = data[required_columns].apply(pd.to_numeric, errors="coerce")
	if values.isna().any().any():
		raise ValueError("Raw dataset contains missing or non-numeric values.")
	if not np.isfinite(values.to_numpy()).all():
		raise ValueError("Raw dataset contains non-finite values.")
	if (values[FEATURE_COLUMNS] < 0).any().any():
		raise ValueError("Feature values cannot be negative.")
	if (values[TARGET_COLUMN] <= 0).any():
		raise ValueError("Salary values must be greater than zero.")

	return {
		"valid": True,
		"rows": int(len(data)),
		"columns": int(len(data.columns)),
		"duplicate_rows": int(data.duplicated().sum()),
	}


def validate_processed_data() -> dict[str, list[int]]:
	paths = {
		"X_train": PROCESSED_DIR / "X_train_final.npy",
		"X_test": PROCESSED_DIR / "X_test_final.npy",
		"y_train": PROCESSED_DIR / "y_train.npy",
		"y_test": PROCESSED_DIR / "y_test.npy",
	}
	missing = [str(path) for path in paths.values() if not path.exists()]
	if missing:
		raise FileNotFoundError(f"Missing processed artifacts: {missing}")

	arrays = {name: np.load(path) for name, path in paths.items()}
	for name, array in arrays.items():
		if array.size == 0 or not np.isfinite(array).all():
			raise ValueError(f"Processed artifact is empty or non-finite: {name}")

	if arrays["X_train"].ndim != 2 or arrays["X_test"].ndim != 2:
		raise ValueError("Processed feature arrays must be two-dimensional.")
	if arrays["X_train"].shape[1] != len(FEATURE_COLUMNS):
		raise ValueError("Processed feature count does not match the schema.")
	if len(arrays["X_train"]) != len(arrays["y_train"]):
		raise ValueError("Training features and targets have different lengths.")
	if len(arrays["X_test"]) != len(arrays["y_test"]):
		raise ValueError("Test features and targets have different lengths.")

	return {name: list(array.shape) for name, array in arrays.items()}


def artifact_manifest() -> dict[str, str]:
	manifest = {}
	for relative_path in ARTIFACTS:
		path = PROJECT_ROOT / relative_path
		if not path.exists():
			raise FileNotFoundError(f"Expected artifact was not created: {path}")
		digest = hashlib.sha256(path.read_bytes()).hexdigest()
		manifest[relative_path] = digest
	return manifest


def run_pipeline(run_number: int, total_runs: int) -> tuple[dict, dict[str, list[int]], dict[str, str]]:
	print(f"\n[INFO] Reproducibility pass {run_number}/{total_runs}", flush=True)
	print("[INFO] Step 1/6: validating raw dataset...", flush=True)
	validation = validate_raw_data()
	print(
		f"[SUCCESS] Raw data valid: {validation['rows']} rows, "
		f"{validation['columns']} columns.",
		flush=True,
	)
	print("[INFO] Step 2/6: preprocessing data...", flush=True)
	preprocess()
	print("[INFO] Step 3/6: validating processed arrays...", flush=True)
	processed_shapes = validate_processed_data()
	print(f"[SUCCESS] Processed data shapes: {processed_shapes}", flush=True)
	print("[INFO] Step 4/6: training regression models...", flush=True)
	train_models()
	print("[INFO] Step 5/6: evaluating models and saving reports...", flush=True)
	evaluate()
	print("[INFO] Step 6/6: hashing output artifacts...", flush=True)
	manifest = artifact_manifest()
	print(f"[SUCCESS] Captured hashes for {len(manifest)} artifacts.", flush=True)
	return validation, processed_shapes, manifest


def main() -> None:
	print("[INFO] Starting Lab 5 reproducibility pipeline.", flush=True)
	print(f"[INFO] Reproducibility seed: {SEED}", flush=True)
	configure_reproducibility()
	OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

	first_validation, processed_shapes, first_manifest = run_pipeline(1, 2)
	_, _, second_manifest = run_pipeline(2, 2)
	print("\n[INFO] Comparing artifact hashes across both passes...", flush=True)
	reproducible = first_manifest == second_manifest

	report = {
		"project": "Salary Prediction",
		"seed": SEED,
		"validation": first_validation,
		"processed_shapes": processed_shapes,
		"reproducible": reproducible,
		"artifact_hashes": first_manifest,
	}
	with (OUTPUTS_DIR / "reproducibility_report.json").open("w", encoding="utf-8") as file:
		json.dump(report, file, indent=2)

	if not reproducible:
		print("[ERROR] Reproducibility check failed.", flush=True)
		raise RuntimeError("Pipeline outputs changed between identical runs.")

	print("[SUCCESS] Reproducibility check passed: artifact hashes match.", flush=True)
	print(f"[SUCCESS] Report saved: {OUTPUTS_DIR / 'reproducibility_report.json'}", flush=True)


if __name__ == "__main__":
	main()
print("\nLab 5 - Data validation and reproducible salary prediction pipeline completed successfully.")