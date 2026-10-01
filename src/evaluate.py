"""
Salary Prediction - Evaluation
------------------------------
Evaluates saved regression models with MAE, MSE, RMSE and R².
Also saves predictions, residuals and simple diagnostic plots.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "R2_percent": float(r2 * 100),
    }


def evaluate():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading test data and trained models...", flush=True)
    X_test = np.load(PROCESSED_DIR / "X_test_final.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    x_test_original = scaler.inverse_transform(X_test).ravel()

    models = {
        "Linear Regression": joblib.load(MODELS_DIR / "linear_regression.pkl"),
        "Polynomial Regression": joblib.load(MODELS_DIR / "polynomial_regression.pkl"),
        "SVR": joblib.load(MODELS_DIR / "svr.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest_baseline.pkl"),
    }

    rows = []
    prediction_store = {}

    for number, (name, model) in enumerate(models.items(), start=1):
        print(f"[INFO] Evaluating model {number}/{len(models)}: {name}...", flush=True)
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        rows.append({"model": name, **metrics})
        prediction_store[name] = pred

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    print(f"[SUCCESS] Model comparison saved to {OUTPUTS_DIR / 'model_comparison.csv'}", flush=True)

    # Final prediction artifact remains based on the registered Random Forest.
    final_pred = prediction_store["Random Forest"]
    print("[INFO] Saving prediction and residual reports...", flush=True)
    np.save(OUTPUTS_DIR / "predictions.npy", final_pred)
    np.save(OUTPUTS_DIR / "salary_predictions.npy", final_pred)

    results = pd.DataFrame({
        "YearsExperience": x_test_original,
        "ActualSalary": y_test,
        "PredictedSalary": final_pred,
        "Residual": y_test - final_pred,
        "AbsoluteError": np.abs(y_test - final_pred),
    }).sort_values("YearsExperience")

    results.to_csv(OUTPUTS_DIR / "actual_vs_predicted.csv", index=False)
    results[["YearsExperience", "Residual", "AbsoluteError"]].to_csv(
        OUTPUTS_DIR / "residuals.csv", index=False
    )

    # Diagnostic plot: actual vs predicted.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, final_pred, alpha=0.8)
    low = min(y_test.min(), final_pred.min())
    high = max(y_test.max(), final_pred.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Random Forest: Actual vs Predicted Salary")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "regression_plot.png", dpi=160)
    plt.close()

    # Diagnostic plot: residuals.
    plt.figure(figsize=(8, 6))
    plt.axhline(0, linestyle="--")
    plt.scatter(x_test_original, y_test - final_pred, alpha=0.8)
    plt.xlabel("Years of Experience")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Random Forest Residuals")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "residual_plot.png", dpi=160)
    plt.close()
    print(f"[SUCCESS] Evaluation reports and plots saved under {OUTPUTS_DIR}", flush=True)

    print("\nModel comparison:")
    for _, row in comparison.iterrows():
        print(f"\n{row['model']}")
        print(f"  MAE:  {row['MAE']:,.2f}")
        print(f"  MSE:  {row['MSE']:,.2f}")
        print(f"  RMSE: {row['RMSE']:,.2f}")
        print(f"  R²:   {row['R2']:.4f} ({row['R2_percent']:.2f}%)")

    print("\nFinal model: Random Forest")
    print(
        f"R²={comparison.loc[comparison['model']=='Random Forest', 'R2'].iloc[0]:.4f}, "
        f"RMSE={comparison.loc[comparison['model']=='Random Forest', 'RMSE'].iloc[0]:,.2f}"
    )


if __name__ == "__main__":
    evaluate()
