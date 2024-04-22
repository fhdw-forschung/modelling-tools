"""Collection of evaluation resources and methods."""

import warnings

import pandas as pd
import plotly.express as px
from pandas import Series
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import root_mean_squared_log_error

PASTEL_ORANGE = "#fa9f55"  # Pastel orange color
PASTEL_BLUE = "#96ceff"  # Pastel blue color


def get_regression_metrics(y_true: Series, y_pred: Series):
    """Get dictionary of common regression metrics.

    Args:
        y_true (``pandas.Series``): The actual values of the ground truth.

        y_pred (``pandas.Series``): The inference values made by the model.
    """
    try:
        rmsle = root_mean_squared_log_error(y_true=y_true, y_pred=y_pred)
    except ValueError:
        rmsle = None

        warnings.warn(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets contain negative values. Therefore it is set to None here."
        )

    metrics = {
        "MAE": mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "MAPE": mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred),
        "RMSE": root_mean_squared_error(y_true=y_true, y_pred=y_pred),
        "RMSLE": rmsle,
        "R2": r2_score(y_true=y_true, y_pred=y_pred),
    }

    return metrics


def plot_estimates(y_true: Series, y_pred: Series, title: str):
    """Plot to compare model inference of a single with actual values.

    Estimates made by the model with ``experiment.predict_model`` are plotted alongside
    with the actual values.

    Args:
        y_true (``pandas.Series``): The actual values of the ground truth.

        y_pred (``pandas.Series``): The inference values made by the model.

        title (``str``): The plot's title.

    Returns:
        A plotly ``Figure`` object. Use its ``show`` function to display it in e.g.
        notebooks.

    Note:
        This function is related to ``plot_model_estimates_multiple_models``. The
        difference is the slightly more convenient notation with explicit ``y_true``
        and ``y_pred`` notation and thereby does demand to define a ``DataFrame``
        in beforehand.
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
        title=title,
        labels={"value": title},
        hover_name=result.index,
        marginal_y="box",
    )
    return figure


def plot_estimates_multiple_models(data: pd.DataFrame, title: str):
    """Plot to compare model inference with actual values.

    Structure of the given DataFrame should be one column per model to be compared.
    The index must be named and is used as the label (i.e. hover information).

    Args:
        data (``pandas.DataFrame``): The ``DataFrame`` with predictions.

        title (``str``): The plot's title.

    Returns:
        A plotly ``Figure`` object. Use its ``show`` function to display it in e.g.
        notebooks.
    """
    figure = px.scatter(
        data.reset_index(),
        x=data.index.name,
        y=list(data.columns),
        title=title,
        hover_name=data.index.name,
        hover_data={data.index.name: False},
        marginal_y="box",
    )
    return figure


def plot_identity(y_true: Series, y_pred: Series, title: str):
    """Plot to compare the predicted output vs. the actual output.

    Args:
        y_true (``pandas.Series``): The Ground Truth. Will be plotted on x-axis.

        y_pred (``pandas.Series``): The predicted values. Will be plotted on y-axis.

        title (``str``): The plot's title.

    Returns:
        A plotly ``Figure`` object. Use its ``show`` function to display it in e.g.
        notebooks.

    Note:
        This function is related to ``plot_identity_multiple_models``. The
        difference is the slightly more convenient notation with explicit ``y_true``
        and ``y_pred`` notation and thereby does demand to define a ``DataFrame``
        in beforehand.
    """
    figure = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "ground truth", "y": "prediction"},
        trendline="ols",
        title=title,
        hover_name=y_true.index,
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


def plot_identity_multiple_models(data: pd.DataFrame, title: str):
    """Plot to compare the predicted output vs. the actual output.

    The ``data``'s structure must include a column named ``y_true``.
    The index must be a named index (is transformed into a ``pandas.Series``).
    All other columns are used as predictions (i.e. models to be compared).

    Args:
        data (``pandas.DataFrame``): The ``DataFrame`` with predictions.

        title (``str``): The plot's title.
    """
    figure = px.scatter(
        data.reset_index(),
        x="y_true",
        y=list(data.columns),
        trendline="ols",
        title=title,
        hover_name=data.index.name,
    )
    y_true = data["y_true"]
    figure.add_shape(
        type="line",
        line={"dash": "dash"},
        x0=y_true.min(),
        y0=y_true.min(),
        x1=y_true.max(),
        y1=y_true.max(),
    )
    return figure


def plot_train_test_target_distribution(target: str, y_train: Series, y_test: Series):
    """Compare the train test distribution of the target variable with a plot.

    Args:
        target (``str``): Name of the learning target. Is used for the plot's title.

        y_train (``pandas.Series``): The target column of the train set.

        y_test (``pandas.Series``): The target column of the test set.
    """
    title = f"Train vs. Test Distribution for target '{target}'."
    data = pd.concat([y_test, y_train], axis=1, keys=["test", "train"])
    fig = px.histogram(
        data,
        marginal="box",
        opacity=0.5,
        nbins=int(len(y_train) * 0.25),
        title=title,
        color_discrete_sequence=[PASTEL_ORANGE, PASTEL_BLUE],
    )
    return fig


def plot_residuals(y_train, pred_train, y_test, pred_test, title):
    """Plot residuals of predictions.

    Args:
        y_train (``pandas.Series``): The target column of the training set.

        pred_train (``pandas.Series``): The prediction of the model on the training set.

        y_test (``pandas.Series``): The target column of the test set.

        pred_test (``pandas.Series``): The prediction of the model on the test set.

        title (``str``): The plot's title.
    """
    result = pd.concat(
        [
            pd.DataFrame(
                {
                    "Prediction": pred_train,
                    "Residual": pred_train - y_train,
                    "Set": "Train",
                }
            ),
            pd.DataFrame(
                {"Prediction": pred_test, "Residual": pred_test - y_test, "Set": "Test"}
            ),
        ]
    )
    figure = px.scatter(
        result,
        x="Prediction",
        y="Residual",
        marginal_y="box",
        color="Set",
        opacity=0.7,
        trendline="ols",
        title=title,
        hover_name=result.index,
    )
    figure.add_hline(y=0, opacity=0.5, line={"dash": "dash"})
    return figure
