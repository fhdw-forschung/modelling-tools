"""Pytest configuration and shared resources."""
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_train_data():
    """Get sample data.

    Returns:
        tuple: the pandas.DataFrame and the name of the target variable.
    """
    return pd.read_csv("fhdw/tests/data/automobile.csv"), "price"
