"""Modelling process resources utilizing pycaret."""
from pathlib import Path

import mlflow
import pandas as pd
from pandas import DataFrame
from pycaret.regression import RegressionExperiment

from fhdw.modelling.base import make_experiment_name


class PyCaretModelManagement:
    """Pycaret Registered Model Management Unit."""

    def __init__(self, model_name: str) -> None:
        """Initialize PyCaret Model management.

        Args:
            model_name (str): The name of the registered model.
        """
        self.name = model_name
        self.client = mlflow.tracking.MlflowClient()

    def _get_parent_run_id(self, stage_name: str):
        model = self._get_model_info_at_stage(stage_name=stage_name)
        run = self.client.get_run(str(model.run_id))
        parent_id = run.data.tags["mlflow.parentRunId"]
        return parent_id

    def _get_data_artifact(self, dataset: str, stage_name: str):
        parent_id = self._get_parent_run_id(stage_name=stage_name)
        data = self.client.download_artifacts(run_id=parent_id, path=f"{dataset}.csv")
        return data

    def _get_model_info_at_stage(self, stage_name: str = "Production"):
        """Get the trained model of the registered model from the given stage."""
        model = self.client.get_latest_versions(name=self.name, stages=[stage_name])

        if len(model) != 1:
            raise ValueError(
                "Retrieved to many search results for versions "
                f"of registered model '{self.name}'; "
                f"expected 1, got {len(model)}"
            )
        return model[0]

    def get_train_data(self, stage_name: str = "Production"):
        """Return the data the named registered model has been trained with."""
        data = self._get_data_artifact(dataset="Train", stage_name=stage_name)
        data = pd.read_csv(data)
        return data

    def get_test_data(self, stage_name: str = "Production"):
        """Return the holdout set used during training of the registered model."""
        data = self._get_data_artifact(dataset="Test", stage_name=stage_name)
        data = pd.read_csv(data)
        return data

    def get_model_at_stage(self, stage_name: str = "Production"):
        """Get the trained model of the registered model from the given stage."""
        reg_path = f"models:/{self.name}/{stage_name}"
        model = mlflow.pyfunc.load_model(reg_path)
        return model


PLOTS = {
    "pipeline": "Schematic drawing of the preprocessing pipeline",
    # "residuals_interactive": "Interactive Residual plots",
    "residuals": "Residuals Plot",
    "error": "Prediction Error Plot",
    # "cooks": "Cooks Distance Plot",
    # "rfe": "Recursive Feat. Selection",
    # "learning": "Learning Curve",
    # "vc": "Validation Curve",
    # "manifold": "Manifold Learning",
    "feature": "Feature Importance",
    # "feature_all": "Feature Importance (All)",
    # "parameter": "Model Hyperparameter",
    # "tree": "Decision Tree",
}


def create_regression_model(
    experiment: RegressionExperiment | None = None,
    data: DataFrame | None = None,
    target: str | None = None,
    exclude: list | None = None,
    include: list | None = None,
    sort_metric: str = "RMSE",
    prefix: str = "",
    save_strategy: str | None = None,
    verbose: int = 0,
    log_experiment: bool = False,
    n_select: int = 3,
    n_iter: int = 25,
    **kwargs,
):
    """Create a regression model with Pycaret.

    This function is a wrapper and convenience function around already quite simplified
    actions defined by pycaret. So, also have a look at the pycaret documentation.

    This convenience functions performs several steps, which strive for the best model
    possible, with minimal configuration provided. Therefore the runtime can be quite
    long. The following pycaret mechanics are progressed (in this order):

    - Create regression experiment.
        Uses ``RegressionExperiment``.
    - Set up regression experiment
        Uses ``setup``.
    - Get the three best performing ML-methods.
        Uses ``compare_models``.
    - Create and tune a model with the best method incl. cross validation.
        Uses ``tune_model``.
    - Create a (single-method) model with standard hyperparameters cross validation.
        Uses ``create_model`` with the best performing ML-method from previous
        ``compare_models``.
    - Create an ensemble with this single-method model.
        *Bagging* procedure.
        Samples a new dataset from the train data with replacement per model.
    - Create an ensemble with this single-method model.
        *Boosting* procedure, which is implemented through
        ``sklearn.ensemble.AdaBoostRegressor``.
    - Create a stacked estimator
        Comprised of the three best methods from comparison.
        The meta-learner is ``LinearRegression``.
    - Create a voting regressor
        Comprised of the three best methods from comparison.
        Trains on the whole dataset.
    - Save artifacts and Input of the process.
        According to ``save_trategy``. Saves to the ``artifacts`` folder which will be
        created if not existing.

    Args:
        train_data (pandas.DataFrame): The training data.

        target (str): The name of the target variable in the train data.

        exclude_models (List[str]): A list of model names to exclude from comparison.
            Cannot be used in conjunction with ``include_models``.

        include_models (List[str]): A list of model names to include in comparison.
            Cannot be used in conjunction with ``exclude_models``.

        sort_metric (str): The metric used to sort the models.

        prefix: A Prefix that will be added to the experiment name. This may help to
            further organize experiments.

        save_strategy (str, optional): The strategy for saving artifacts. When
            ``"local"``, save in local ``artifacts`` folder. Defaults to ``None``,
            i.e. save nothing. This does not affect tracking with ``mlflow``, incl.
            model logging through an artifact store. This depends on ``log_experiment``
            option.

        verbose (int, optional): Whether to print training output. This affects all
            training steps. Values in ``[0, 1, 2]``, where ``0`` is non-verbous, ``1``
            is minimal information and ``2`` prints the full pipeline information.

        log_experiment (bool, optional): Whether to log via MLflow. Activates logs for
            experiment, data and plots.

        n_select (int, optional): Numer of methods to be selected from the method
            comparison. Selected methods will be incorporated in ensembles. Higher
            numbers increase the runtime of the function significantly! When
            ``n_select=1``, no ensembles are built and evaluated.

        n_iter (int): Number of parameter settings that are sampled. ``n_iter`` trades
            off runtime vs quality of the solution. In PyCaret tuning is implemented
            through ``sklearn.model_selection.RandomizedSearchCV``.
            See scikit-learn documentation for details.

    Returns:
        ``tuple``: The ``RegressionExperiment`` and the best trained model model.
    """
    min_sel = 1  # at least select one method
    verb_levels = [0, 1, 2]

    if exclude and include:
        raise ValueError("Cannot use both 'include' and 'exclude'.")
    if n_select < min_sel:
        raise ValueError(f"`n_select` too low, must be at least {min_sel}.")
    if include and len(include) < n_select:
        raise ValueError("When using include, provide at least `n_select` choices.")
    if verbose not in verb_levels:
        raise ValueError(f"verbose must have levels in '{verb_levels}'")

    verbose_pycaret = verbose == 2  # pycaret expects boolean; highest verbosity

    if isinstance(experiment, RegressionExperiment) and data is None and target is None:
        exp = experiment
    elif experiment is None and isinstance(target, str) and isinstance(data, DataFrame):
        exp_name = make_experiment_name(target=target, prefix=prefix)
        log_plots = list(PLOTS.keys()) if log_experiment else log_experiment
        if verbose:
            print(f"experiment name: '{exp_name}'")
        _print_info("experiment setup...", level=verbose)
        exp = RegressionExperiment()
        exp.setup(
            data=data,
            target=target,
            experiment_name=exp_name,
            verbose=verbose_pycaret,
            log_experiment=log_experiment,
            log_data=log_experiment,
            log_plots=log_plots,
            **kwargs,
        )
    else:
        raise ValueError("Either provide pre-defined experiment OR data and target.")

    _print_info("compare models...", level=verbose)

    best_methods = exp.compare_models(
        exclude=exclude,
        include=include,
        sort=sort_metric,
        n_select=n_select,
        verbose=verbose_pycaret,
    )

    # take into account that best_methods is not a list if `n_select=1`
    best = best_methods[0] if n_select > min_sel else best_methods

    _print_info("tune model...", level=verbose)
    tuned = exp.tune_model(
        estimator=best,
        choose_better=True,
        optimize=sort_metric,
        n_iter=n_iter,
        verbose=verbose_pycaret,
    )

    if n_select > min_sel:
        _print_info("ensemble model (Bagging)...", level=verbose)
        exp.ensemble_model(
            estimator=tuned,
            choose_better=False,
            method="Bagging",
            verbose=verbose_pycaret,
        )
        try:
            _print_info("ensemble model (Boosting)...", level=verbose)
            exp.ensemble_model(
                estimator=tuned,
                choose_better=False,
                method="Boosting",
                verbose=verbose_pycaret,
            )
        except TypeError:
            print(f"Skipped ensemble with Boosting. Estimator {tuned} not supported.")

        _print_info("stack models...", level=verbose)
        exp.stack_models(
            estimator_list=best_methods,
            choose_better=False,
            restack=False,
            verbose=verbose_pycaret,
        )

        _print_info("blend models (Voting)...", level=verbose)
        exp.blend_models(
            estimator_list=best_methods,
            choose_better=False,
            optimize=sort_metric,
            verbose=verbose_pycaret,
        )

    best_model = exp.automl(optimize=sort_metric, turbo=False)

    if save_strategy == "local":
        # saving artifacts
        path_e = persist_experiment(experiment=exp, strategy=save_strategy)
        path_d = persist_data(experiment=exp, strategy=save_strategy)
        path_m = persist_model(experiment=exp, model=best_model, strategy=save_strategy)
        if verbose:
            print(f"saved experiment to: '{path_e}'")
            print(f"saved data to: '{path_d}'")
            print(f"saved best model to: '{path_m}.pkl'")
    elif save_strategy is not None:
        raise ValueError("unknown saving strategy")

    return exp, best_model


def get_model_paths(folder: str = "artifacts/models", stategy: str = "local"):
    """Retrieves a list of model files from the specified folder and subfolders.

    Recursive ``Path.glob``.

    Args:
        folder (``str``, optional): Path to the folder containing model files. Defaults
            to ``"models"``. Folder will be created if not existing.

        file_extension (str, optional): File extension for model files. Defaults to
            ``"pkl"``.

        strategy (``str``, optional): Retrieval strategy. Currently, only ``"local"``
            strategy is supported. Other strategies like MLflow might be supported in
            the future.

    Returns:
        ``List[Path]``: A list of ``Path`` objects representing the model files in the
        specified folder.

    Raises:
        ``NotADirectoryError``: If the specified folder does not exist or is not a
            directory.

        ``ValueError``: If an unsupported retrieval strategy is specified.
    """
    if not Path(folder).is_dir():
        raise NotADirectoryError(f"'{folder}' either not existing or not a folder.")

    if stategy == "local":
        return list(Path(folder).glob("**/*.pkl"))

    raise ValueError("unknown saving strategy")


def persist_data(
    experiment: RegressionExperiment,
    folder: str = "artifacts/data",
    strategy: str = "local",
):
    """Persists the dataset from a ``RegressionExperiment``.

    Args:
        experiment (``RegressionExperiment``): The experiment containing the dataset.

        folder (``str``, optional): The folder path to save the dataset. Defaults to
            ``"experiments/data"``. Folder will be created if not existing.

        strategy (``str``, optional): The strategy for saving the dataset. Defaults to
            ``"local"``.

    Returns:
        ``str``: The path where the dataset is saved.

    Raises:
        ``ValueError``: Raised when an unknown saving strategy is provided.

    Example:
        ::

            experiment = RegressionExperiment(...)
            persist_data(experiment, folder="custom_folder", strategy="local")

    """
    data: DataFrame = experiment.dataset

    if strategy == "local":
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = f"{folder}/{experiment.exp_name_log}.parquet"
        data.to_parquet(path)
        return path

    raise ValueError("unknown saving strategy")


def persist_experiment(
    experiment: RegressionExperiment,
    folder: str = "artifacts/experiments",
    strategy: str = "local",
):
    """Persist the given experiment.

    Saves the experiment with all configuration. The data must be saved separately.
    You could use ``persist_data`` for this.

    Args:
        experiment (``RegressionExperiment``): The experiment to be persisted.

        folder (``str``, optional): The folder path where the experiment will be saved.
            Defaults to ``"experiments"``.

        strategy (``str``, optional): The saving strategy. Currently, only ``"local"``
            strategy is supported. Defaults to ``"local"``.

    Returns:
        ``str``: The path where the experiment is saved.

    Raises:
        ``ValueError``: Raised when an unknown saving strategy is provided.

    Note:
        This function is a convenience wrapper for ``exp.save_experiment`` to simplify
        boilerplate code.

    Example:
        ::

            persist_experiment(
                my_regression_exp, folder="saved_experiments", strategy="local"
            )
            ``'saved_experiments/my_experiment_log.pkl'``
    """
    if strategy == "local":
        exp_folder = Path(folder)
        exp_folder.mkdir(parents=True, exist_ok=True)
        path_exp = f"{exp_folder}/{experiment.exp_name_log}.exp"
        experiment.save_experiment(path_or_file=path_exp)
        return path_exp

    raise ValueError("unknown saving strategy")


def persist_model(
    experiment: RegressionExperiment,
    model,
    folder: str = "artifacts/models",
    strategy: str = "local",
):
    """Persist the given model.

    Args:
        experiment (``RegressionExperiment``): The regression experiment object.

        model: The trained model to be persisted.

        folder (``str``, optional): The folder where the model will be saved.
            Defaults to ``"models"``.

        strategy (``str``, optional): The saving strategy.
            Currently, only ``"local"`` strategy is supported. Defaults to ``"local"``.

    Returns:
        ``pathlib.Path``: The path where the model is saved.

    Raises:
        ``ValueError``: If an unknown saving strategy is provided.

    Note:
        This function is a convenience wrapper around the ``save_model`` method
        of the provided ``experiment`` object. It automatically manages the
        boilerplate code for saving the model with the appropriate name derived
        from the experiment.
    """
    if strategy == "local":
        model_folder = Path(folder)
        model_folder.mkdir(parents=True, exist_ok=True)
        path_model = model_folder / Path(experiment.exp_name_log)
        experiment.save_model(model=model, model_name=str(path_model))
        return path_model

    raise ValueError("unknown saving strategy")


def _print_info(text: str, level):
    """Print depending on the verbosity level."""
    if level == 1:  # verbosity 2 not considered, pycaret does this
        print(text)
