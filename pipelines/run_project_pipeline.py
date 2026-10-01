"""Run the complete salary prediction MLOps workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "salary.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage(number: int, total: int, lab_name: str, script_name: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"[INFO] Stage {number}/{total}: {lab_name}", flush=True)
    print(f"[INFO] Starting {script_name}", flush=True)
    print("=" * 72, flush=True)
    subprocess.run(
        [sys.executable, "-u", str(PROJECT_ROOT / "pipelines" / script_name)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"[SUCCESS] Stage {number}/{total}: {lab_name} completed.", flush=True)


def main() -> None:
    print("===========================================================", flush=True)
    print("Salary Prediction - Complete MLOps Pipeline")
    print("===========================================================", flush=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-lab5",
        "--skip-reproducibility",
        dest="skip_lab5",
        action="store_true",
        help="Skip Lab 5 validation and its two-pass reproducibility check (legacy alias: --skip-reproducibility).",
    )
    args = parser.parse_args()

    stages = [
        ("Lab 3 - Baseline Regression", "run_lab3_baseline.py"),
        ("Lab 4 - MLflow Experiment Tracking", "run_lab4_tracking.py"),
    ]
    if not args.skip_lab5:
        stages.append(("Lab 5 - Data Validation and Reproducibility", "run_lab5_baseline.py"))
    stages.append(("Lab 6 - Model Registry and Versioning", "run_lab6_baseline.py"))

    print(f"\n[INFO] Running {len(stages)} lab stages in order...", flush=True)
    for number, (lab_name, script_name) in enumerate(stages, start=1):
        run_stage(number, len(stages), lab_name, script_name)

    print("\n[INFO] Finalizing run manifest...", flush=True)
    report_paths = {"registry": OUTPUTS_DIR / "model_registry_report.json"}
    if not args.skip_lab5:
        report_paths["validation"] = OUTPUTS_DIR / "reproducibility_report.json"
    missing = [str(path) for path in report_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Pipeline reports were not created: {missing}")

    registry_report = json.loads(report_paths["registry"].read_text(encoding="utf-8"))
    manifest = {
        "pipeline": "salary_prediction_mlops",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": "data/raw/salary.csv",
            "sha256": sha256(RAW_DATASET),
        },
        "stages": [lab_name for lab_name, _ in stages],
        "registered_model": {
            "name": registry_report["model_name"],
            "version": registry_report["version"],
            "uri": registry_report["model_uri"],
            "metrics": registry_report["metrics"],
        },
        "reports": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in report_paths.items()},
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUTS_DIR / "pipeline_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[SUCCESS] Complete MLOps pipeline finished.")
    print(f"Registered champion: {registry_report['model_uri']}")
    print(f"Run manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()