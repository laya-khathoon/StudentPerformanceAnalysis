from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def get_classification_models(y_train=None):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    # Calculate class imbalance weight for XGBoost
    if y_train is not None:
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()

        scale_weight = (
            negative_count / positive_count
            if positive_count > 0 else 1.0
        )
    else:
        scale_weight = 1.0

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    return models
