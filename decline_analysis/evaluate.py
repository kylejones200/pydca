"""
Evaluation metrics for decline curve analysis forecasts.
"""

import numpy as np
import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(numerator / denominator) * 100)


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """R-squared coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle constant values case where ss_tot = 0
    if ss_tot == 0:
        # If actual values are constant and predictions match, R² = 1
        if ss_res == 0:
            return 1.0
        # If actual values are constant but predictions don't match, R² = 0
        else:
            return 0.0

    return float(1 - (ss_res / ss_tot))


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Comprehensive evaluation of forecast performance."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
