"""Tracking Resources."""

import mlflow
from pandas import Series

from fhdw.modelling.evaluation import get_regression_metrics


def log_metrics_to_mlflow(y_true: Series, y_pred: Series, prefix: str = ""):
    """Log metrics to active MLflow Experiment and Run.

    Args:
        y_true (``pandas.Series``): The ground truth values.

        y_pred (``pandas.Series``): The values made by model inference.

        prefix (``str``): Prefix for the metric names. This could be used to specify the
            metrics' purpose. Example: When set to *test_* the resulting metrics are
            e.g. *test_RMSE*.
    """
    prefix = f"{prefix}_" if prefix and prefix[-1] != "_" else prefix

    metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)

    # filter out none values, because it is not allowed to log None values to mlflow
    metrics = {key: value for key, value in metrics.items() if value is not None}

    metrics = {f"{prefix}{metric}": v for metric, v in metrics.items()}
    mlflow.log_metrics(metrics=metrics)
    return True
