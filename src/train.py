"""
Salary Prediction - Model Training
----------------------------------
Trains two regression baselines:
1. Linear Regression
2. Polynomial Regression
3. Support Vector Regression
4. Random Forest Regressor

Artifacts are stored under models/.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def train_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(PROCESSED_DIR / "X_train_final.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")

    models = [
        ("Linear Regression", "linear_regression.pkl", LinearRegression()),
        ("Polynomial Regression", "polynomial_regression.pkl", make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        )),
        ("SVR", "svr.pkl", TransformedTargetRegressor(
            regressor=SVR(kernel="rbf", C=10.0),
            transformer=StandardScaler(),
        )),
        ("Random Forest", "random_forest_baseline.pkl", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )),
    ]

    print(f"[INFO] Training {len(models)} regression models...", flush=True)
    for number, (model_name, artifact_name, model) in enumerate(models, start=1):
        print(f"[INFO] Model {number}/{len(models)}: training {model_name}...", flush=True)
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / artifact_name)
        print(f"[SUCCESS] {model_name} saved to {MODELS_DIR / artifact_name}", flush=True)

    # Keep the standalone tracking pipeline's artifact aligned with training.
    random_forest = next(model for name, _, model in models if name == "Random Forest")
    joblib.dump(random_forest, MODELS_DIR / "random_forest_mlflow.pkl")
    (MODELS_DIR / "decision_tree_baseline.pkl").unlink(missing_ok=True)

    print("[SUCCESS] All regression models trained.", flush=True)


if __name__ == "__main__":
    train_models()
