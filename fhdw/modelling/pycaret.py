"""Modelling resources utiliying pycaret for the process."""
from pathlib import Path

from pandas import DataFrame
from pycaret.regression import RegressionExperiment

from fhdw.modelling.base import make_experiment_name
from fhdw.modelling.base import validate_path


def create_regression_model(
    train_data: DataFrame,
    target: str,
    prefix: str = "",
    sort_metric: str = "RMSE",
    exclude: list | None = None,
    include: list | None = None,
):
    """Create a regression model with Pycaret.

    This function is a wrapper and convenience function around already quite simplified
    actions defined by pycaret. So, also have a look at the pycaret documentation.
    Following pycaret mechanics are progressed through this function (in this order):
    - create regression experiment (`RegressionExperiment`)
    - set up regression experiment (`setup`)
    - get best model (`compare_models`)
    - create model with standard hyperparameters cross validation (`create_model`)
    - tune model with cross validation (`tune_model`)
    - final training including test data (`finalize_model`)
    - the final model is saved to the `models` folder (`save_model`), which will be
    created if not existing

    Args:
        train_data: The training data.

        target: The name of the target variable in the train data.

        prefix: A Prefix that will be added to all names that are given in the process.
        This is e.g. the experiment name or the name of the model that is saved locally.

        sort_metric (str): The metric used to sort the models.

        exclude_models (List[str]): A list of model names to exclude from comparison.
        Cannot be used in conjunction with `include_models`.

        include_models (List[str]): A list of model names to include in comparison.
        Cannot be used in conjunction with `exclude_models`.

    Returns:
        tuple: The RegressionExperiment and the trained Pipeline containing the model.
    """
    if exclude and include:
        raise ValueError("Cannot use both 'include' and 'exclude'.")

    exp_name = make_experiment_name(target=target, prefix=prefix)
    print(f"experiment name: '{exp_name}'")

    # experiment setup
    exp = RegressionExperiment()
    exp.setup(data=train_data, target=target, experiment_name=exp_name)

    # model creation with picking best model and tuning, up to finalization
    best_method = exp.compare_models(exclude=exclude, include=include, sort=sort_metric)
    trained_model = exp.create_model(best_method)
    tuned_model = exp.tune_model(trained_model, choose_better=True)
    finalized_model = exp.finalize_model(tuned_model)

    persist_model(experiment=exp, model=finalized_model)

    return exp, finalized_model


def persist_data(
    experiment: RegressionExperiment,
    folder: str = "experiments/data",
    strategy: str = "local",
):
    """Persists the dataset from a RegressionExperiment.

    Args:
        experiment (RegressionExperiment): The experiment containing the dataset.

        folder (str, optional): The folder path to save the dataset. Defaults to
        "experiments/data". Folder will be created if not existing.

        strategy (str, optional): The strategy for saving the dataset. Defaults to
        "local".

    Returns:
        str: The path where the dataset is saved.

    Raises:
        ValueError: Raised when an unknown saving strategy is provided.

    Example:
        experiment = RegressionExperiment(...)
        persist_data(experiment, folder="custom_folder", strategy="local")
    """
    data: DataFrame = experiment.dataset

    if strategy == "local":
        Path(folder).mkdir(exist_ok=True)
        path = f"{folder}/{experiment.exp_name_log}.parquet"
        data.to_parquet(path)
        return path

    raise ValueError("unknown saving strategy")


def persist_model(
    experiment: RegressionExperiment,
    model,
    folder: str = "models",
    strategy: str = "local",
):
    """Persist the given model.

    Args:
        experiment (RegressionExperiment): The regression experiment object.

        model: The trained model to be persisted.

        folder (str, optional): The folder where the model will be saved.
        Defaults to "models".

        strategy (str, optional): The saving strategy.
        Currently, only "local" strategy is supported. Defaults to "local".

    Returns:
        Path: The path where the model is saved.

    Raises:
        ValueError: If an unknown saving strategy is provided.

    Note:
        This function is a convenience wrapper around the `save_model` method
        of the provided `experiment` object. It automatically manages the
        boilerplate code for saving the model with the appropriate name derived
        from the experiment.
    """
    if strategy == "local":
        model_folder = Path(folder)
        model_folder.mkdir(exist_ok=True)
        path_model = model_folder / Path(experiment.exp_name_log)
        experiment.save_model(model=model, model_name=str(path_model))
        return path_model

    raise ValueError("unknown saving strategy")


def get_model_paths(folder: str = "models", stategy: str = "local"):
    """Retrieves a list of model files from the specified folder and subfolders.

    Recursive `Path.glob`.

    Args:
        folder (str, optional): Path to the folder containing model files. Defaults
        to "models". Folder will be created if not existing.

        file_extension (str, optional): File extension for model files. Defaults to
        "pkl".

        strategy (str, optional): Retrieval strategy. Currently, only "local" strategy
        is supported. Other strategies like MLFlow might be supported in the future.

    Returns:
        List[Path]: A list of Path objects representing the model files in the specified
        folder.

    Raises:
        NotADirectoryError: If the specified folder does not exist or is not a
        directory.

        ValueError: If an unsupported retrieval strategy is specified.
    """
    if not validate_path(folder):
        raise NotADirectoryError(f"'{folder}' either not existing or not a folder.")

    if stategy == "local":
        return list(Path(folder).glob("**/*.pkl"))

    raise ValueError("unknown saving strategy")


def persist_experiment(
    experiment: RegressionExperiment,
    folder: str = "experiments",
    strategy: str = "local",
):
    """Persist the given experiment.

    Saves the experiment with all configuration. The data must be saved separately.
    You could use `persist_data` for this.

    Args:
        experiment (RegressionExperiment): The experiment to be persisted.

        folder (str, optional): The folder path where the experiment will be saved.
        Defaults to "experiments".

        strategy (str, optional): The saving strategy. Currently, only "local"
        strategy is supported. Defaults to "local".

    Returns:
        str: The path where the experiment is saved.

    Raises:
        ValueError: Raised when an unknown saving strategy is provided.

    Note:
        This function is a convenience wrapper for `exp.save_experiment` to simplify
        boilerplate code.

    Example:
        >>> persist_experiment(
                my_regression_exp, folder="saved_experiments", strategy="local"
            )
        'saved_experiments/my_experiment_log.pkl'
    """
    if strategy == "local":
        exp_folder = Path(folder)
        exp_folder.mkdir(exist_ok=True)
        path_exp = f"{exp_folder}/{experiment.exp_name_log}.exp"
        experiment.save_experiment(path_or_file=path_exp)
        return path_exp

    raise ValueError("unknown saving strategy")
