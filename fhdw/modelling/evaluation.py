"""Collection of evaluation resources and methods."""

import warnings

import pandas as pd
import plotly.express as px
from pandas import Series
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score


def get_regression_metrics(y_true, y_pred):
    """Get dictionary of common regression metrics.

    Args:
        y_true: The actual values of the ground truth.

        y_pred: The inference values made by the model.
    """
    try:
        rmsle = mean_squared_log_error(y_true=y_true, y_pred=y_pred, squared=False)
    except ValueError:
        rmsle = None

        warnings.warn(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets contain negative values. Therefore it is set to None here."
        )

    metrics = {
        "MAE": mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "MAPE": mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred),
        "RMSE": mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
        "RMSLE": rmsle,
        "R2": r2_score(y_true=y_true, y_pred=y_pred),
    }

    return metrics


def plot_model_estimates(y_true, y_pred, target: str):
    """Plot to compare model inference with actual values.

    Estimates made by the model with `experiment.predict_model` are plotted alongside
    with the actual values.

    Args:
        y_true: The actual values of the ground truth.

        y_pred: The inference values made by the model.

        target (string): The learning target. Will be used for titles and labels.
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


def plot_identity(y_true: Series, y_pred: Series, target: str):
    """Plot to compare the predicted output vs. the actual output.

    Args:
        y_true: The Ground Truth. Will be plotted on x-axis.

        y_pred: The predicted values. Will be plotted on y-axis.

        target: will be used for the plots title.
    """
    figure = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "ground truth", "y": "prediction"},
        title=target,
        trendline="ols",
    )
    figure.add_shape(
        type="line",
        line={"dash": "dash"},
        x0=y_true.min(),
        y0=y_true.min(),
        x1=y_true.max(),
        y1=y_true.max(),
    )
    return figure
