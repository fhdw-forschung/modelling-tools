"""Tracking Resources."""

import mlflow

from fhdw.modelling.evaluation import get_regression_metrics


def log_metrics_to_mlflow(y_true, y_pred, prefix: str = ""):
    """Log metrics to active MLflow Experiment and Run."""
    prefix = f"{prefix}_" if prefix and prefix[-1] != "_" else prefix

    metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)
    metrics = {f"{prefix}{metric}": v for metric, v in metrics.items()}
    mlflow.log_metrics(metrics=metrics)
    return True
