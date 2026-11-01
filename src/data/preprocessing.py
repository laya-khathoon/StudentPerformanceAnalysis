
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """
    Build preprocessing pipelines for numerical and categorical features.
    """

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def prepare_data(
    df,
    target="G3",
    include_period_grades=True,
    test_size=0.2,
    random_state=42
):
    """
    Split the dataset and preprocess training and testing features.

    include_period_grades=True:
        Uses G1 and G2 for later-stage performance prediction.

    include_period_grades=False:
        Excludes G1 and G2 for early performance prediction.
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    if not include_period_grades:
        X = X.drop(
            columns=["G1", "G2"],
            errors="ignore"
        )

    # Split before fitting any preprocessing transformations
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Build and fit preprocessing using training data only
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )
