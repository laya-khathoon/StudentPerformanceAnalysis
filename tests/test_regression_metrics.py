import pytest

from src.models.evaluation import evaluate_regression


def test_perfect_prediction_has_zero_error_and_r2_one():
    result = evaluate_regression([10, 12, 14], [10, 12, 14])
    assert result["MAE"] == 0
    assert result["RMSE"] == 0
    assert result["R2 Score"] == pytest.approx(1.0)


def test_constant_offset_gives_matching_mae_and_rmse():
    result = evaluate_regression([10, 12, 14], [11, 13, 15])
    assert result["MAE"] == pytest.approx(1.0)
    assert result["RMSE"] == pytest.approx(1.0)
    assert result["MSE"] == pytest.approx(1.0)


def test_rmse_is_square_root_of_mse():
    result = evaluate_regression([0, 0], [3, 4])
    assert result["RMSE"] == pytest.approx(result["MSE"] ** 0.5)
