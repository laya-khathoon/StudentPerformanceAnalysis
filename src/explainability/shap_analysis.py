from pathlib import Path

import joblib
import shap
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.data.loader import load_school_datasets
from src.data.preprocessing import prepare_data


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "school" / "classification"
FIGURE_DIR = ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def generate_shap_explanations():

    math_df, portuguese_df = load_school_datasets()

    datasets = {
        "Mathematics": math_df,
        "Portuguese": portuguese_df
    }

    for subject, df in datasets.items():

        for mode in ["early", "later"]:

            print(f"\nExplaining: {subject} | {mode}")

            X_train, X_test, y_train, y_test, preprocessor = prepare_data(
                df,
                prediction_mode=mode,
                random_state=42
            )

            model_path = (
                MODEL_DIR /
                f"{subject.lower()}_{mode}_best_model.joblib"
            )

            model = joblib.load(model_path)

            # Convert transformed data to dense arrays.
            X_background = X_train[:100]
            X_sample = X_test[:100]

            if hasattr(X_background, "toarray"):
                X_background = X_background.toarray()

            if hasattr(X_sample, "toarray"):
                X_sample = X_sample.toarray()

            feature_names = preprocessor.get_feature_names_out()

            # Select the appropriate SHAP explainer.
            if isinstance(
                model,
                (
                    DecisionTreeClassifier,
                    RandomForestClassifier,
                    XGBClassifier
                )
            ):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

            elif isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(
                    model,
                    X_background
                )
                shap_values = explainer.shap_values(X_sample)

            else:
                raise TypeError(
                    f"Unsupported model type: {type(model)}"
                )

            # For binary classification, explain class 1:
            # At Risk.
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]

            prefix = f"{subject.lower()}_{mode}"

            # SHAP summary plot
            plt.figure(figsize=(10, 7))

            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                max_display=15
            )

            plt.title(
                f"{subject} - {mode.title()} - SHAP Summary"
            )

            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR / f"{prefix}_shap_summary.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            # SHAP feature importance bar plot
            plt.figure(figsize=(10, 7))

            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
                max_display=15
            )

            plt.title(
                f"{subject} - {mode.title()} - SHAP Importance"
            )

            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR / f"{prefix}_shap_importance.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(f"SHAP plots saved for {subject} | {mode}")

    print("\nSHAP EXPLAINABILITY COMPLETED")


if __name__ == "__main__":
    generate_shap_explanations()
