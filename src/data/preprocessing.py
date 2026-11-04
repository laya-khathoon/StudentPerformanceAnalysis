
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineering import create_features


def build_preprocessor(X):
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

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
    prediction_mode="later",
    test_size=0.2,
    random_state=42
):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    X_train = create_features(X_train, prediction_mode=prediction_mode)
    X_test = create_features(X_test, prediction_mode=prediction_mode)

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
