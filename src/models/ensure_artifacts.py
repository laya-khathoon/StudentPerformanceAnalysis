"""Ensure dataset and trained models exist (used by hosted deployments).

The GitHub repo intentionally does not store the dataset or the trained
model files. On a fresh deployment (for example Streamlit Community
Cloud) they are rebuilt once on first use: the UCI Student Performance
dataset is downloaded and the training scripts are run. Everything is
idempotent - if the files already exist, nothing happens.
"""

from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00320/student.zip"
)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "school"
DATA_FILES = ["student-mat.csv", "student-por.csv"]

MODEL_DIRS = [
    PROJECT_ROOT / "models" / "school" / "regression",
    PROJECT_ROOT / "models" / "school" / "classification",
]

SUBJECTS = ["mathematics", "portuguese"]
MODES = ["early", "later"]
SUFFIXES = ["best_model.joblib", "preprocessor.joblib"]


def _data_missing():
    return any(not (DATA_DIR / name).exists() for name in DATA_FILES)


def _models_missing():
    for directory in MODEL_DIRS:
        for subject in SUBJECTS:
            for mode in MODES:
                for suffix in SUFFIXES:
                    name = f"{subject}_{mode}_{suffix}"
                    if not (directory / name).exists():
                        return True
    return False


def _download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "student.zip"
    print("Downloading Student Performance dataset from UCI...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for name in DATA_FILES:
            archive.extract(name, DATA_DIR)
    zip_path.unlink()


def _train_models():
    for module in (
        "src.models.train_regression",
        "src.models.train_classification",
    ):
        print(f"Training models with {module} (one-time setup)...")
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_ROOT,
            check=True,
        )


def ensure_artifacts():
    """Download the dataset and train models if they are missing."""
    if _data_missing():
        _download_data()
    if _models_missing():
        _train_models()


def ensure_data():
    """Download the dataset if it is missing."""
    if _data_missing():
        _download_data()
