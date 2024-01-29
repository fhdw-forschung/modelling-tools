"""Preprocessing Tests."""
import numpy as np
import pandas as pd
import pytest

from fhdw.modelling.preprocessing import apply_isolation_forest
from fhdw.modelling.preprocessing import remove_outliers_isolation


@pytest.fixture(name="test_data")
def sample_data():
    """Create a sample DataFrame for testing."""
    data = {
        "Feature1": [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
        "Feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_remove_outliers_isolation(test_data):
    """Simple test; Ensure the function does not raise an exception."""
    remove_outliers_isolation(test_data)


def test_apply_isolation_forest(test_data):
    """Simple test; Ensure the function does not raise an exception."""
    apply_isolation_forest(test_data)
