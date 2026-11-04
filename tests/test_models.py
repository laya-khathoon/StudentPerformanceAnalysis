import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.models.classification import get_classification_models
from src.models.regression import get_regression_models


def test_classification_model_names():
    models = get_classification_models()
    assert set(models.keys()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
    }


def test_classification_model_types():
    models = get_classification_models()

    assert isinstance(models["Logistic Regression"], LogisticRegression)
    assert isinstance(models["Decision Tree"], DecisionTreeClassifier)
    assert isinstance(models["Random Forest"], RandomForestClassifier)
    assert isinstance(models["XGBoost"], XGBClassifier)


def test_xgboost_class_weight_calculation():
    y_train = np.array([0, 0, 0, 0, 1, 1])
    models = get_classification_models(y_train)

    assert models["XGBoost"].get_params()["scale_pos_weight"] == 2.0


def test_classification_models_handle_no_positive_samples():
    y_train = np.array([0, 0, 0, 0])
    models = get_classification_models(y_train)

    assert models["XGBoost"].get_params()["scale_pos_weight"] == 1.0


def test_regression_model_names():
    models = get_regression_models()

    assert set(models.keys()) == {
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
    }


def test_regression_model_types():
    models = get_regression_models()

    assert isinstance(models["Linear Regression"], LinearRegression)
    assert isinstance(models["Decision Tree"], DecisionTreeRegressor)
    assert isinstance(models["Random Forest"], RandomForestRegressor)
    assert isinstance(models["XGBoost"], XGBRegressor)
