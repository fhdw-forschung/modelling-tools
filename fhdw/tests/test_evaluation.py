"""Test evaluation resources."""
import pandas as pd

from fhdw.modelling.evaluation import get_regression_metrics
from fhdw.modelling.evaluation import plot_identity
from fhdw.modelling.evaluation import plot_model_estimates


def test_get_regression_metrics_positive():
    """Test regression metrics calculation for positive values."""
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, 2.2, 2.9, 4.2, 5.1])

    metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)

    assert len(metrics) == 5
    assert metrics["RMSLE"] is not None


def test_get_regression_metrics_negative():
    """Test regression metrics calculation with negative values."""
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, -2.2, 2.9, 4.2, 5.1])

    metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)

    assert len(metrics) == 5
    assert metrics["RMSLE"] is None

    y_true = pd.Series([1, 2, -3, 4, 5])
    y_pred = pd.Series([1.1, 2.2, 2.9, 4.2, 5.1])

    metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)

    assert len(metrics) == 5
    assert metrics["RMSLE"] is None

    y_true = pd.Series([-1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, -2.2, 2.9, 4.2, 5.1])

    metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)

    assert len(metrics) == 5
    assert metrics["RMSLE"] is None


def test_plot_estimates_model_vs_actual():
    """Test vs-plot properties after generating the plot."""
    # Mock data for testing
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.2, 2.8, 3.7, 4.9]
    target_name = "Test Target"

    # Call the function to generate the plot
    figure = plot_model_estimates(y_true, y_pred, target_name)

    assert figure.layout.title.text == target_name  # type: ignore
    assert figure.layout.xaxis.title.text == "index"  # type: ignore
    assert figure.layout.yaxis.title.text == target_name  # type: ignore

    # Check if the data in the plot matches the input data
    assert figure.data[0].x.tolist() == list(range(len(y_true)))
    assert figure.data[0].y.tolist() == y_pred
    assert figure.data[2].y.tolist() == y_true
    assert len(figure.data) == 4


def test_plot_actual_vs_pred():
    """Test identity-plot properties after generating the plot."""
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, 2.2, 2.9, 4.2, 5.1])
    target = "Test Plot"

    figure = plot_identity(y_true, y_pred, target)

    figure_dict = figure.to_dict()
    assert figure_dict["layout"]["title"]["text"] == target
    assert figure_dict["layout"]["xaxis"]["title"]["text"] == "ground truth"
    assert figure_dict["layout"]["yaxis"]["title"]["text"] == "prediction"
    assert len(figure_dict["layout"]["shapes"]) == 1
    assert figure_dict["layout"]["shapes"][0]["type"] == "line"
