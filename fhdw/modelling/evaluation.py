"""Collection of evaluation resources and methods."""

import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score


def get_regression_metrics(y_true, y_pred):
    """Get dictionary of common regression metrics."""
    metrics = {
        "MAE": mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "MAPE": mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred),
        "RMSE": mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
        "RMSLE": mean_squared_log_error(y_true=y_true, y_pred=y_pred, squared=False),
        "R2": r2_score(y_true=y_true, y_pred=y_pred),
    }
    return metrics


def plot_estimates_model_vs_actual(y_true, y_pred, target: str):
    """Plot to compare estimates.

    Estimates made by the model with `experiment.predict_model` are plotted alongside
    with the actual values.

    Args:
        y_true: The actual values of the ground truth.

        y_pred: The inference values made by the model.

        target: The learning target. Will be used for titles and labels.
    """
    result = pd.DataFrame(
        {
            "Model": y_pred,
            "y_true": y_true,
        }
    )
    figure = px.scatter(
        result,
        x=result.index,
        y=["Model", "y_true"],
        title=target,
        labels={"value": target},
        hover_name=result.index,
        marginal_y="box",
    )
    return figure
