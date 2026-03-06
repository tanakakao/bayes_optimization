import pandas as pd
import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def reg_scores(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> pd.DataFrame:
    """回帰評価指標 (RMSE, MAE, MAPE, R2)。"""
    return pd.DataFrame(
        {
            "RMSE": [root_mean_squared_error(y_true, y_pred)],
            "MAE": [mean_absolute_error(y_true, y_pred)],
            "MAPE": [mean_absolute_percentage_error(y_true, y_pred)],
            "R2": [r2_score(y_true, y_pred)],
        },
        index=[target],
    )


def cls_scores(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> pd.DataFrame:
    """分類評価指標 (ACCURACY, PRECISION, RECALL, F1)。"""
    return pd.DataFrame(
        {
            "ACCURACY": [accuracy_score(y_true, y_pred)],
            "PRECISION": [precision_score(y_true, y_pred)],
            "RECALL": [recall_score(y_true, y_pred)],
            "F1": [f1_score(y_true, y_pred)],
        },
        index=[target],
    )
