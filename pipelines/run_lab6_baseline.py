"""Lab 6 - Register and version the salary prediction model with MLflow."""
from __future__ import annotations

print("=====================================================")
print("Executing Lab 6 - Model Registry and Versioning for Salary Prediction")
print("=====================================================")

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess
from src.train import train_models

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MODEL_NAME = "salary-random-forest"
MODEL_ALIAS = "champion"


def calculate_metrics(model) -> dict[str, float]:
	X_test = np.load(PROCESSED_DIR / "X_test_final.npy")
	y_test = np.load(PROCESSED_DIR / "y_test.npy")
	predictions = model.predict(X_test)
	mse = mean_squared_error(y_test, predictions)
	return {
		"mae": float(mean_absolute_error(y_test, predictions)),
		"mse": float(mse),
		"rmse": float(np.sqrt(mse)),
		"r2": float(r2_score(y_test, predictions)),
	}


def main() -> None:
	print("[INFO] Starting Lab 6 model registration pipeline.", flush=True)
	try:
		import mlflow
		import mlflow.sklearn
		from mlflow.tracking import MlflowClient
	except ImportError as exc:
		raise SystemExit(
			"MLflow is required. Install project dependencies with: "
			"pip install -r requirements.txt"
		) from exc

	print("[INFO] Step 1/6: preparing processed data...", flush=True)
	preprocess()
	print("[SUCCESS] Processed data is ready.", flush=True)
	print("[INFO] Step 2/6: training regression models...", flush=True)
	train_models()
	print("[SUCCESS] Regression models are ready.", flush=True)

	print("[INFO] Step 3/6: loading Random Forest and calculating metrics...", flush=True)
	model_path = MODELS_DIR / "random_forest_mlflow.pkl"
	if not model_path.exists():
		raise FileNotFoundError(f"Model artifact was not created: {model_path}")
	model = joblib.load(model_path)
	metrics = calculate_metrics(model)

	tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment("Salary Prediction")

	print("[INFO] Step 4/6: logging run and registering model with MLflow...", flush=True)
	with mlflow.start_run(run_name="RandomForestRegistry") as run:
		mlflow.log_params(
			{
				"model_type": "RandomForestRegressor",
				"n_estimators": 200,
				"max_depth": 10,
				"min_samples_leaf": 2,
				"random_state": 42,
				"n_jobs": 1,
			}
		)
		mlflow.log_metrics(metrics)
		mlflow.set_tags(
			{
				"pipeline": "lab6_model_registry",
				"dataset": "data/raw/salary.csv",
				"lifecycle": "candidate",
			}
		)
		mlflow.sklearn.log_model(
			model,
			name="random_forest_model",
			registered_model_name=MODEL_NAME,
			skops_trusted_types=["sklearn.tree._tree.Tree"],
		)
		run_id = run.info.run_id

	client = MlflowClient()
	versions = client.search_model_versions(f"name='{MODEL_NAME}'")
	run_versions = [version for version in versions if version.run_id == run_id]
	if not run_versions:
		raise RuntimeError("MLflow registered no model version for the new run.")

	print("[INFO] Step 5/6: assigning candidate and champion aliases...", flush=True)
	version = max(run_versions, key=lambda item: int(item.version))
	client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
	client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)
	client.set_model_version_tag(MODEL_NAME, version.version, "validated", "true")
	client.update_model_version(
		name=MODEL_NAME,
		version=version.version,
		description=(
			"Validated Random Forest baseline. "
			f"R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}."
		),
	)

	registered_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
	print("[INFO] Step 6/6: loading champion and verifying predictions...", flush=True)
	champion = mlflow.sklearn.load_model(registered_uri)
	expected = np.load(PROCESSED_DIR / "X_test_final.npy")
	expected_predictions = model.predict(expected)
	champion_predictions = champion.predict(expected)
	np.testing.assert_allclose(expected_predictions, champion_predictions)

	report = {
		"model_name": MODEL_NAME,
		"version": int(version.version),
		"run_id": run_id,
		"aliases": {"candidate": version.version, MODEL_ALIAS: version.version},
		"model_uri": registered_uri,
		"tracking_uri": tracking_uri,
		"metrics": metrics,
		"prediction_verification": "passed",
	}
	
	OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
	report_path = OUTPUTS_DIR / "model_registry_report.json"
	print("[INFO] Writing model registry report...", flush=True)
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

	print(f"[SUCCESS] Registered model: {MODEL_NAME}, version {version.version}", flush=True)
	print(f"[SUCCESS] Champion URI: {registered_uri}", flush=True)
	print("[SUCCESS] Registry prediction verification passed.", flush=True)
	print(f"[SUCCESS] Report saved: {report_path}", flush=True)


if __name__ == "__main__":
	main()
print("\nLab 6 - Model Registry and Versioning for Salary Prediction completed successfully.")