from pathlib import Path
import subprocess
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT/"models"/"loan_approval_pipeline.joblib"

@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    if not MODEL_PATH.exists():
        subprocess.run(
            [sys.executable, "-m", "src.train"],
            cwd=PROJECT_ROOT,
            check=True,
        )

@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT
