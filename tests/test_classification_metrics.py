import pytest

from src.models.evaluation import evaluate_classification


def test_confusion_counts_are_correct():
    result = evaluate_classification([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.4, 0.2, 0.6])
    assert (result["TP"], result["FN"], result["TN"], result["FP"]) == (1, 1, 1, 1)
    assert result["Accuracy"] == pytest.approx(0.5)


def test_no_positive_predictions_gives_zero_precision_not_an_error():
    result = evaluate_classification([1, 0], [0, 0], [0.2, 0.1])
    assert result["Precision"] == 0
    assert result["Recall"] == 0
    assert result["F1 Score"] == 0


def test_perfect_ranking_gives_auc_one():
    result = evaluate_classification([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["ROC-AUC"] == pytest.approx(1.0)
