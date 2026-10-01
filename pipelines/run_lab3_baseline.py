"""
Lab 3 - Baseline Salary Regression Pipeline

Run from the project root:
    python pipelines/run_lab3_baseline.py
"""
print("====================================================")
print("Executing Lab 3 - Baseline Salary Regression Pipeline")
print("=====================================================")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess
from src.train import train_models
from src.evaluate import evaluate


def main():
    print("=" * 70)
    print("SALARY PREDICTION - BASELINE PIPELINE")
    print("=" * 70)

    print("\n[1/3] Preprocessing")
    preprocess()

    print("\n[2/3] Training")
    train_models()

    print("\n[3/3] Evaluation")
    evaluate()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
print("\nLab 3 - Baseline Salary Regression Pipeline completed successfully.")