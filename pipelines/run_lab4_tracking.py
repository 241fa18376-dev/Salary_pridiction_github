"""
Lab 4 - MLflow Experiment Tracking for Salary Prediction

Run from the project root:
    python pipelines/run_lab4_tracking.py

The script saves a local model artifact and logs parameters/metrics to MLflow.
"""
print("==================================================================")
print("Executing Lab 4 - MLflow Experiment Tracking for Salary Prediction")
print("==================================================================")
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def main():
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise SystemExit(
            "MLflow is required for this pipeline. Install it with: pip install mlflow"
        ) from exc

    # Make sure preprocessing artifacts exist.
    preprocess()

    X_train = np.load(PROCESSED_DIR / "X_train_final.npy")
    X_test = np.load(PROCESSED_DIR / "X_test_final.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "random_state": 42,
    }

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"],
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, pred),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "random_forest_mlflow.pkl")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("Salary Prediction")

    with mlflow.start_run(run_name="RandomForestRegressor") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    print("MLflow tracking completed.")
    print(f"Experiment: {mlflow.get_experiment_by_name('Salary Prediction').name}")
    print(f"Run ID: {run.info.run_id}")
    print("Metrics:", {k: round(float(v), 4) for k, v in metrics.items()})
    print(f"Saved: {MODELS_DIR / 'random_forest_mlflow.pkl'}")


if __name__ == "__main__":
    main()
print("\nLab 4 - MLflow Experiment Tracking for Salary Prediction completed successfully.")