"""Test cases for the pycaret modelling tools."""

from pycaret.internal.pipeline import Pipeline
from pycaret.regression import RegressionExperiment

from fhdw.modelling.pycaret import create_regression_model


# Basic test case with minimum required inputs
def test_create_regression_model_minimal(sample_train_data):
    """Basic test case with minimum required inputs."""
    train_data = sample_train_data[0]
    target = sample_train_data[1]
    exp, model = create_regression_model(
        train_data,
        target,
        include=["knn"],
    )
    print(type(model))

    assert isinstance(exp, RegressionExperiment)
    assert isinstance(model, Pipeline)
