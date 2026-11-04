import pytest

from src.models.evaluation import (
    evaluate_regression,
    evaluate_classification
)


def test_regression_metrics():
    y_true = [10, 20, 30, 40]
    y_pred = [12, 18, 28, 42]

    result = evaluate_regression(y_true, y_pred)

    assert result["MAE"] == pytest.approx(2.0)
    assert result["MSE"] == pytest.approx(4.0)
    assert result["RMSE"] == pytest.approx(2.0)
    assert result["R2 Score"] == pytest.approx(0.968)


def test_regression_perfect_predictions():
    y_true = [10, 20, 30, 40]

    result = evaluate_regression(y_true, y_true)

    assert result["MAE"] == 0
    assert result["MSE"] == 0
    assert result["RMSE"] == 0
    assert result["R2 Score"] == pytest.approx(1.0)


def test_classification_metrics():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.7, 0.9, 0.4, 0.8]

    result = evaluate_classification(y_true, y_pred, y_prob)

    assert result["Accuracy"] == pytest.approx(4 / 6)
    assert result["Precision"] == pytest.approx(2 / 3)
    assert result["Recall"] == pytest.approx(2 / 3)
    assert result["F1 Score"] == pytest.approx(2 / 3)
    assert result["ROC-AUC"] == pytest.approx(8 / 9)


def test_classification_confusion_matrix():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.7, 0.9, 0.4, 0.8]

    result = evaluate_classification(y_true, y_pred, y_prob)

    assert result["TN"] == 2
    assert result["FP"] == 1
    assert result["FN"] == 1
    assert result["TP"] == 2


def test_classification_zero_division():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    y_prob = [0.1, 0.2, 0.3, 0.4]

    result = evaluate_classification(y_true, y_pred, y_prob)

    assert result["Precision"] == 0
    assert result["Recall"] == 0
    assert result["F1 Score"] == 0

