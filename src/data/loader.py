
from pathlib import Path
import pandas as pd


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "school"


def load_school_datasets():
    """
    Load Mathematics and Portuguese student datasets.
    """

    math_path = DATA_DIR / "student-mat.csv"
    portuguese_path = DATA_DIR / "student-por.csv"

    if not math_path.exists():
        raise FileNotFoundError(f"Missing dataset: {math_path}")

    if not portuguese_path.exists():
        raise FileNotFoundError(f"Missing dataset: {portuguese_path}")

    math_df = pd.read_csv(math_path, sep=";")
    portuguese_df = pd.read_csv(portuguese_path, sep=";")

    return math_df, portuguese_df


def validate_dataset(df, dataset_name):
    """
    Validate the basic structure and quality of a dataset.
    """

    required_columns = [
        "school",
        "age",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "G3"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")

    print(f"\n{dataset_name} validation")
    print("-" * 40)
    print("Records:", len(df))
    print("Features:", len(df.columns))
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    return True
