from pathlib import Path

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from src.data.loader import load_school_datasets
from src.data.preprocessing import prepare_data


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "school" / "classification"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

CLASSIFICATION_RESULTS = REPORT_DIR / "classification_results.csv"
REGRESSION_RESULTS = REPORT_DIR / "regression_results.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def generate_classification_plots():

    math_df, portuguese_df = load_school_datasets()

    datasets = {
        "Mathematics": math_df,
        "Portuguese": portuguese_df
    }

    results = pd.read_csv(CLASSIFICATION_RESULTS)

    best_models = (
        results.sort_values("F1 Score", ascending=False)
        .groupby(["Subject", "Prediction Mode"])
        .first()
        .reset_index()
    )

    for _, row in best_models.iterrows():

        subject = row["Subject"]
        mode = row["Prediction Mode"]
        model_name = row["Model"]

        print(f"Generating plots: {subject} | {mode}")

        df = datasets[subject]

        X_train, X_test, y_train, y_test, preprocessor = prepare_data(
            df,
            prediction_mode=mode,
            random_state=42
        )

        y_test_risk = (y_test < 10).astype(int)

        model_path = (
            MODEL_DIR /
            f"{subject.lower()}_{mode}_best_model.joblib"
        )

        model = joblib.load(model_path)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        prefix = f"{subject.lower()}_{mode}"

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_test_risk,
            predictions,
            labels=[0, 1],
            display_labels=["Not At Risk", "At Risk"],
            cmap="Blues",
            ax=ax
        )

        ax.set_title(f"{subject} - {mode.title()} - {model_name}")
        fig.tight_layout()

        fig.savefig(
            FIGURE_DIR / f"{prefix}_confusion_matrix.png",
            dpi=300
        )
        plt.close(fig)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test_risk, probabilities)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {roc_auc:.3f})"
        )

        ax.plot([0, 1], [0, 1], linestyle="--")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{subject} - {mode.title()} ROC Curve")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            FIGURE_DIR / f"{prefix}_roc_curve.png",
            dpi=300
        )
        plt.close(fig)


def generate_regression_plots():

    results = pd.read_csv(REGRESSION_RESULTS)

    for subject in results["Subject"].unique():

        for mode in results["Prediction Mode"].unique():

            subset = results[
                (results["Subject"] == subject) &
                (results["Prediction Mode"] == mode)
            ].copy()

            if subset.empty:
                continue

            subset = subset.sort_values("RMSE")

            prefix = f"{subject.lower()}_{mode}"

            # RMSE comparison
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(subset["Model"], subset["RMSE"])
            ax.set_title(f"{subject} - {mode.title()} - RMSE")
            ax.set_ylabel("RMSE")
            ax.set_xlabel("Regression Model")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()

            fig.savefig(
                FIGURE_DIR / f"{prefix}_rmse_comparison.png",
                dpi=300
            )
            plt.close(fig)

            # R2 comparison
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(subset["Model"], subset["R2 Score"])
            ax.set_title(f"{subject} - {mode.title()} - R2 Score")
            ax.set_ylabel("R2 Score")
            ax.set_xlabel("Regression Model")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()

            fig.savefig(
                FIGURE_DIR / f"{prefix}_r2_comparison.png",
                dpi=300
            )
            plt.close(fig)


if __name__ == "__main__":

    print("Generating classification evaluation graphs...")
    generate_classification_plots()

    print("\nGenerating regression comparison graphs...")
    generate_regression_plots()

    print("\nEVALUATION GRAPHS GENERATED SUCCESSFULLY")
    print("Output directory:", FIGURE_DIR)
