"""Tracking resources tests."""

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from fhdw.modelling.evaluation import get_regression_metrics
from fhdw.modelling.tracking import ModelManagement
from fhdw.modelling.tracking import log_metrics_to_mlflow


@pytest.fixture(name="fix_data")
def sample_data():
    """Fixture with sample data."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 2.2, 2.8, 4.2]
    return y_true, y_pred


@pytest.fixture(name="mock_pycaret_model_management")
def pycaret_model_management():
    """Create PycaretModelManagement Object."""
    mm = ModelManagement(model_name="example_model")
    mm.client = MagicMock()
    return mm


def test_log_metrics_to_mlflow(fix_data):
    """Test case for log_metrics_to_mlflow function."""
    y_true, y_pred = fix_data
    mock_log_metrics = MagicMock()

    # Patch mlflow.log_metrics with mock object
    with patch("mlflow.log_metrics", mock_log_metrics):
        log_metrics_to_mlflow(y_true, y_pred)
        expected_metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)
        mock_log_metrics.assert_called_once_with(metrics=expected_metrics)


def test_log_metrics_to_mlflow_negative_value_in_prediction():
    """Test case for log_metrics_to_mlflow function.

    This test is to ensure that the mlflow log function is not called with a None value.
    A None value is included when one of the predictions is negative, which results in
    the fact, that the RMSLE cannot be calculated.
    """
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.0, -2.0, 3.0, 4.0])
    mock_log_metrics = MagicMock()

    # Patch mlflow.log_metrics with mock object
    with patch("mlflow.log_metrics", mock_log_metrics):
        log_metrics_to_mlflow(y_true, y_pred)
        expected_metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)
        expected_metrics.pop("RMSLE")
        mock_log_metrics.assert_called_once_with(metrics=expected_metrics)


def test_log_metrics_to_mlflow_prefix(fix_data):
    """Test case for log_metrics_to_mlflow function."""
    y_true, y_pred = fix_data
    mock_log_metrics = MagicMock()

    # Patch mlflow.log_metrics with the mock object
    with patch("mlflow.log_metrics", mock_log_metrics):
        log_metrics_to_mlflow(y_true, y_pred, prefix="test")
        without_separator = get_regression_metrics(y_true=y_true, y_pred=y_pred)
        without_separator = {f"test_{k}": v for k, v in without_separator.items()}
        mock_log_metrics.assert_called_with(metrics=without_separator)

        log_metrics_to_mlflow(y_true, y_pred, prefix="sup_")
        with_separator = get_regression_metrics(y_true=y_true, y_pred=y_pred)
        with_separator = {f"sup_{k}": v for k, v in with_separator.items()}
        mock_log_metrics.assert_called_with(metrics=with_separator)


@patch("mlflow.pyfunc.load_model")
def test_get_model_at_stage(mock_load_model, mock_pycaret_model_management):
    """Test model retrieval procedure."""
    # Mocking mlflow.pyfunc.load_model
    mock_model = Mock()
    mock_load_model.return_value = mock_model

    model = mock_pycaret_model_management.get_model_at_stage()

    mock_load_model.assert_called_once_with("models:/example_model/Production")
    assert model == mock_model
