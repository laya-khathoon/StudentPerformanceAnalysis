from pathlib import Path

import joblib
import pandas as pd

from src.data.loader import load_school_datasets
from src.data.preprocessing import prepare_data
from src.models.regression import get_regression_models
from src.models.evaluation import evaluate_regression


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "school" / "regression"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def train_regression_models():

    math_df, portuguese_df = load_school_datasets()

    datasets = {
        "Mathematics": math_df,
        "Portuguese": portuguese_df
    }

    results = []

    for subject, df in datasets.items():

        for mode in ["early", "later"]:

            print(f"\nTraining: {subject} | {mode}")

            X_train, X_test, y_train, y_test, preprocessor = prepare_data(
                df,
                prediction_mode=mode,
                random_state=42
            )

            models = get_regression_models()

            best_rmse = float("inf")
            best_model_name = None

            for model_name, model in models.items():

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                metrics = evaluate_regression(y_test, predictions)

                print(f"\n{model_name}")

                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")

                results.append({
                    "Subject": subject,
                    "Prediction Mode": mode,
                    "Model": model_name,
                    **metrics
                })

                if metrics["RMSE"] < best_rmse:

                    best_rmse = metrics["RMSE"]
                    best_model_name = model_name

                    joblib.dump(
                        model,
                        MODEL_DIR / f"{subject.lower()}_{mode}_best_model.joblib"
                    )

                    joblib.dump(
                        preprocessor,
                        MODEL_DIR / f"{subject.lower()}_{mode}_preprocessor.joblib"
                    )

            print(f"\nBest model: {best_model_name}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "regression_results.csv",
        index=False
    )

    print("\nREGRESSION TRAINING COMPLETED")

    print(
        results_df.sort_values(
            ["Subject", "Prediction Mode", "RMSE"]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    train_regression_models()
