# Salary Prediction MLOps Workflow

This project demonstrates a small but complete machine-learning lifecycle.

## Run the full project

From the project root:

```powershell
& ".\.venv\Scripts\python.exe" ".\pipelines\run_project_pipeline.py"
```

The runner executes the labs in order and prints a numbered banner and completion status for each:

1. **Lab 3 - Baseline regression:** preprocessing, training, and model comparison.
2. **Lab 4 - MLflow experiment tracking:** logs model parameters, metrics, and the model artifact.
3. **Lab 5 - Data validation and reproducibility:** validates raw and processed data and compares artifact hashes across two runs.
4. **Lab 6 - Model registry and versioning:** registers the model, assigns `candidate` and `champion` aliases, loads the champion, and verifies predictions.

The full run writes `outputs/pipeline_run_manifest.json`. It records the dataset hash, registered model name and version, model URI, metrics, and completed stages. MLflow metadata is stored in `mlflow.db`.

## Run individual lifecycle steps

```powershell
& ".\.venv\Scripts\python.exe" ".\pipelines\run_lab3_baseline.py"
& ".\.venv\Scripts\python.exe" ".\pipelines\run_lab4_tracking.py"
& ".\.venv\Scripts\python.exe" ".\pipelines\run_lab5_baseline.py"
& ".\.venv\Scripts\python.exe" ".\pipelines\run_lab6_baseline.py"
```

Use `--skip-lab5` to skip Lab 5 (including its data validation and reproducibility checks):

```powershell
& ".\.venv\Scripts\python.exe" ".\pipelines\run_project_pipeline.py" --skip-lab5
```

## MLOps concepts shown

- **Data quality:** invalid input stops the pipeline before training.
- **Lineage:** dataset and output reports identify what produced a model version.
- **Reproducibility:** fixed seeds, single-threaded training, and artifact hash comparison.
- **Experiment tracking:** parameters and metrics are logged to MLflow.
- **Model registry:** each registration creates a version under `salary-random-forest`.
- **Model promotion:** aliases separate a candidate version from the current champion.
- **Deployment confidence:** the champion is loaded from the registry and its predictions are verified before the run succeeds.