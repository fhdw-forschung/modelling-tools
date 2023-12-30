"""Test cases for the pycaret modelling tools."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pycaret.regression import RegressionExperiment
from pycaret.regression import load_experiment
from sklearn.linear_model._coordinate_descent import ElasticNet
from sklearn.linear_model._ridge import Ridge
from sklearn.neighbors._regression import KNeighborsRegressor

from fhdw.modelling.pycaret import create_regression_model
from fhdw.modelling.pycaret import get_model_paths
from fhdw.modelling.pycaret import persist_data
from fhdw.modelling.pycaret import persist_experiment


@pytest.fixture(scope="session", name="experiment")
def dummy_experiment(sample_train_data):
    """Run once per training Session."""
    exp_path = Path("artifacts/experiments/dummy_experiment.pkl")
    train_data = sample_train_data[0]

    if not exp_path.exists():
        exp = RegressionExperiment()
        target = sample_train_data[1]
        exp.setup(data=train_data, target=target, experiment_name=str(exp_path.stem))
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        exp.save_experiment(exp_path)
    else:
        exp = load_experiment(path_or_file=exp_path, data=train_data)

    return exp


@pytest.fixture(name="mock_experiment")
def mock_regression_experiment():
    """Create Mock object with RegressionExperiment notation."""
    mock = MagicMock(spec=RegressionExperiment)
    mock.exp_name_log = "test_experiment"
    return mock


# Basic test case with minimum required inputs
def test_create_regression_model_minimal(sample_train_data):
    """Basic test case with minimum required inputs."""
    train_data = sample_train_data[0]
    target = sample_train_data[1]
    exp, model = create_regression_model(
        data=train_data,
        target=target,
        include=["knn", "en", "ridge"],
        prefix="test",
    )
    assert type(model) in (ElasticNet, Ridge, KNeighborsRegressor)
    assert isinstance(exp, RegressionExperiment)


def test_create_experiment_with_kwargs(sample_train_data):
    """Check if setup of experiment with kwargs sets configuration correctly."""
    train_data = sample_train_data[0]
    target = sample_train_data[1]
    exp, _ = create_regression_model(
        data=train_data,
        target=target,
        include=["knn", "en", "ridge"],
        transform_target=True,  # not defined in func-params, hence becomes kwarg
        prefix="test",
    )
    config_tt = exp.get_config("transform_target_param")
    assert config_tt is True


def test_create_experiment_with_single_selection_n_select(sample_train_data):
    """Test single method selection to check that no ensembles are built."""
    train_data = sample_train_data[0]
    target = sample_train_data[1]
    exp, model = create_regression_model(
        data=train_data,
        target=target,
        include=["knn", "en", "ridge"],
        transform_target=True,
        prefix="test",
        n_select=1,
    )
    assert isinstance(model, (ElasticNet, Ridge, KNeighborsRegressor))
    assert isinstance(exp, RegressionExperiment)


def test_persist_data_unknown_strategy(experiment):
    """Test model persistence with unknown strategy.

    should raise Notimplemented.
    """
    with pytest.raises(ValueError, match="unknown saving strategy"):
        persist_data(experiment=experiment, strategy="unknownlol", folder="")


def test_persist_data_explicit_notation(experiment, tmp_path):
    """Test model persistence with unknown strategy.

    should raise Notimplemented.
    """
    result = persist_data(experiment=experiment, strategy="local", folder=str(tmp_path))
    assert isinstance(result, str)
    assert Path(result).exists()


def test_get_model_paths_custom_valid_folder(tmp_path):
    """Test get_model_paths with a custom folder."""
    result = get_model_paths(folder=tmp_path)  # temp folder as custom folder
    assert result == list(Path(tmp_path).glob("**/*.pkl"))


def test_get_model_paths_custom_strategy_valid_path(tmp_path):
    """Test get_model_paths with a custom retrieval strategy."""
    custom_strategy = "mlflow"
    with pytest.raises(ValueError, match="unknown saving strategy"):
        get_model_paths(folder=tmp_path, stategy=custom_strategy)


def test_get_model_paths_invalid_folder():
    """Test get_model_paths with an invalid folder."""
    with pytest.raises(
        NotADirectoryError,
        match="'invalid_folder' either not existing or not a folder.",
    ):
        get_model_paths(folder="invalid_folder")


def test_persist_experiment_local_strategy(mock_experiment, tmp_path):
    """Test experiment persist in legal scenario."""
    # Arrange
    experiment = mock_experiment
    folder = tmp_path
    strategy = "local"

    # Act
    result = persist_experiment(experiment, folder, strategy)

    # Assert
    assert result == f"{folder}/{experiment.exp_name_log}.exp"
    experiment.save_experiment.assert_called_once_with(path_or_file=result)


def test_persist_experiment_unknown_strategy(mock_experiment):
    """Test experiment persist with unknown saving strategy."""
    # Arrange
    experiment = mock_experiment
    folder = "experiments"
    strategy = "unknown_strategy"

    # Act & Assert
    with pytest.raises(ValueError, match="unknown saving strategy"):
        persist_experiment(experiment, folder, strategy)
