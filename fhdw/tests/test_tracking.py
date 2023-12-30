"""Tracking resources tests."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from fhdw.modelling.evaluation import get_regression_metrics
from fhdw.modelling.tracking import log_metrics_to_mlflow


@pytest.fixture(name="fix_data")
def sample_data():
    """Fixture with sample data."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 2.2, 2.8, 4.2]
    return y_true, y_pred


def test_log_metrics_to_mlflow(fix_data):
    """Test case for log_metrics_to_mlflow function."""
    y_true, y_pred = fix_data

    # Create a MagicMock for mlflow.log_metrics
    mock_log_metrics = MagicMock()

    # Patch mlflow.log_metrics with the mock object
    with patch("mlflow.log_metrics", mock_log_metrics):
        # Call the function under test
        log_metrics_to_mlflow(y_true, y_pred)

        # Assert that mlflow.log_metrics is called with the expected arguments
        expected_metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)
        mock_log_metrics.assert_called_once_with(metrics=expected_metrics)
