from pathlib import Path

import joblib
import pandas as pd

from src.data.loader import load_school_datasets
from src.data.preprocessing import prepare_data
from src.models.classification import get_classification_models
from src.models.evaluation import evaluate_classification


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "school" / "classification"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def train_classification_models():

    math_df, portuguese_df = load_school_datasets()

    datasets = {
        "Mathematics": math_df,
        "Portuguese": portuguese_df
    }

    results = []

    for subject, df in datasets.items():

        for mode in ["early", "later"]:

            print(f"\n{'=' * 60}")
            print(f"Training: {subject} | {mode}")
            print("=" * 60)

            X_train, X_test, y_train, y_test, preprocessor = prepare_data(
                df,
                prediction_mode=mode,
                random_state=42
            )

            # Positive class (1) means the student is at risk.
            y_train_risk = (y_train < 10).astype(int)
            y_test_risk = (y_test < 10).astype(int)

            models = get_classification_models(y_train_risk)

            best_f1 = -1
            best_model_name = None

            for model_name, model in models.items():

                model.fit(X_train, y_train_risk)

                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)[:, 1]

                metrics = evaluate_classification(
                    y_test_risk,
                    predictions,
                    probabilities
                )

                print(f"\n{model_name}")

                for metric, value in metrics.items():
                    if metric in ["TN", "FP", "FN", "TP"]:
                        print(f"{metric}: {int(value)}")
                    else:
                        print(f"{metric}: {value:.4f}")

                results.append({
                    "Subject": subject,
                    "Prediction Mode": mode,
                    "Model": model_name,
                    **metrics
                })

                # Select the best model using F1 score.
                if metrics["F1 Score"] > best_f1:

                    best_f1 = metrics["F1 Score"]
                    best_model_name = model_name

                    joblib.dump(
                        model,
                        MODEL_DIR / (
                            f"{subject.lower()}_{mode}_best_model.joblib"
                        )
                    )

                    joblib.dump(
                        preprocessor,
                        MODEL_DIR / (
                            f"{subject.lower()}_{mode}_preprocessor.joblib"
                        )
                    )

            print(f"\nBest model: {best_model_name}")
            print(f"Best F1 Score: {best_f1:.4f}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "classification_results.csv",
        index=False
    )

    print("\nCLASSIFICATION TRAINING COMPLETED")

    print(
        results_df.sort_values(
            ["Subject", "Prediction Mode", "F1 Score"],
            ascending=[True, True, False]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    train_classification_models()
