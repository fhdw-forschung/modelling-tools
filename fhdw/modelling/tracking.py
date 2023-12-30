"""Tracking Resources.

Tracking could be implemented with MLflow for example. Functions that utilize tracking
with MLflow will be named accordingly.
"""

import mlflow

from fhdw.modelling.evaluation import get_regression_metrics


def log_metrics_to_mlflow(y_true, y_pred):
    """Log metrics to active MLflow Experiment and Run."""
    mlflow.log_metrics(metrics=get_regression_metrics(y_true=y_true, y_pred=y_pred))
    return True
