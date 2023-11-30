"""Test cases for the pycaret modelling tools."""
from pathlib import Path

import pytest
from pycaret.internal.pipeline import Pipeline
from pycaret.regression import RegressionExperiment
from pycaret.regression import load_experiment

from fhdw.modelling.pycaret import create_regression_model
from fhdw.modelling.pycaret import persist_data


@pytest.fixture(scope="session", name="experiment")
def dummy_experiment(sample_train_data):
    """Run once per training Session."""
    exp_path = Path("experiments/dummy_experiment.pkl")
    train_data = sample_train_data[0]

    if not exp_path.exists():
        exp = RegressionExperiment()
        target = sample_train_data[1]
        exp.setup(data=train_data, target=target, experiment_name=str(exp_path.stem))
        exp_path.parent.mkdir(exist_ok=True)
        exp.save_experiment(exp_path)
    else:
        exp = load_experiment(path_or_file=exp_path, data=train_data)

    return exp


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


def test_persist_data_unknown_strategy(experiment):
    """Test model persistence with unknown strategy.

    should raise Notimplemented.
    """
    with pytest.raises(NotImplementedError):
        persist_data(experiment=experiment, strategy="unknownlol", folder="")


def test_persist_data_explicit_notation(experiment, tmp_path):
    """Test model persistence with unknown strategy.

    should raise Notimplemented.
    """
    result = persist_data(experiment=experiment, strategy="local", folder=str(tmp_path))
    assert isinstance(result, str)
    assert Path(result).exists()
