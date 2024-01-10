"""Modelling process resources utilizing pycaret."""
from pathlib import Path

from pandas import DataFrame
from pycaret.regression import RegressionExperiment

from fhdw.modelling.base import make_experiment_name

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
    verbose: bool = False,
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
    - create regression experiment (`RegressionExperiment`)
    - set up regression experiment (`setup`)
    - get the three best performing ML-methods (`compare_models`)
    - create and tune a model with the best method incl. cross validation (`tune_model`)
    - create a (single-method) model with standard hyperparameters cross validation
    (`create_model`) with the best performing ML-method from previous `compare_models`
    - create an ensemble with this single-method model (bagging procedure);
    samples a new dataset from the train data with replacement per model
    - create an ensemble with this single-method model (boosting procedure);
    Boosting is implemented through `sklearn.ensemble.AdaBoostRegressor`
    - create a stacked estimator comprised of the three best methods from comparison;
    the meta-learner is `LinearRegression`
    - create a voting regressor comprised of the three best methods from comparison;
    trains on the whole dataset

    - artifacts and input of the process will be saved (optionally, `save_strategy`) to
    the `artifacts` folder which will be created if not existing.

    Args:
        train_data: The training data.

        target: The name of the target variable in the train data.

        exclude_models (List[str]): A list of model names to exclude from comparison.
        Cannot be used in conjunction with `include_models`.

        include_models (List[str]): A list of model names to include in comparison.
        Cannot be used in conjunction with `exclude_models`.

        sort_metric (str): The metric used to sort the models.

        prefix: A Prefix that will be added to the experiment name. This may help to
        further organize experiments.

        save_strategy (str, optional): The strategy for saving artifacts. When "local",
        save in local `artifacts` folder. Defaults to `None`, i.e. save nothing.
        This does not affect tracking with `mlflow`, incl. model logging through an
        artifact store. This depends on `log_experiment` option.

        verbose (bool, optional): Whether to print training output. This affects all
        training steps.

        log_experiment (bool, optional): Whether to log via MLflow. Activates logs for
        experiment, data and plots.

        n_select (int, optional): Numer of methods to be selected from the method
        comparison. Selected methods will be incorporated in ensembles. Higher numbers
        increase the runtime of the function significantly! When `n_select=1`, no
        ensembles are built and evaluated.

        n_iter (int): Number of parameter settings that are sampled. `n_iter` trades
        off runtime vs quality of the solution. In PyCaret tuning is implemented through
        `sklearn.model_selection.RandomizedSearchCV`.
        See scikit-learn documentation for details.

    Returns:
        The `RegressionExperiment` and the tuned Pipeline containing the model.
    """
    min_sel = 1  # at least select one method

    if exclude and include:
        raise ValueError("Cannot use both 'include' and 'exclude'.")
    if n_select < min_sel:
        raise ValueError(f"`n_select` too low, must be at least {min_sel}.")
    if include and len(include) < n_select:
        raise ValueError("When using include, provide at least `n_select` choices.")

    if isinstance(experiment, RegressionExperiment):
        exp = experiment
    elif experiment is None and isinstance(target, str) and isinstance(data, DataFrame):
        exp_name = make_experiment_name(target=target, prefix=prefix)
        log_plots = list(PLOTS.keys()) if log_experiment else log_experiment
        if verbose:
            print(f"experiment name: '{exp_name}'")
        exp = RegressionExperiment()
        exp.setup(
            data=data,
            target=target,
            experiment_name=exp_name,
            verbose=verbose,
            log_experiment=log_experiment,
            log_data=log_experiment,
            log_plots=log_plots,
            **kwargs,
        )
    else:
        raise ValueError("Either provide pre-defined experiment OR data and target.")

    # model tuning with best method
    best_methods = exp.compare_models(
        exclude=exclude,
        include=include,
        sort=sort_metric,
        n_select=n_select,
        verbose=verbose,
    )

    # take into account that best_methods is not a list if `n_select=1`
    best = best_methods[0] if n_select > min_sel else best_methods

    tuned = exp.tune_model(
        best, choose_better=True, optimize=sort_metric, n_iter=n_iter, verbose=verbose
    )

    if n_select > min_sel:
        # ensemble best methods, after creating the initial model
        exp.ensemble_model(
            estimator=tuned, choose_better=False, method="Bagging", verbose=verbose
        )
        try:
            exp.ensemble_model(
                estimator=tuned, choose_better=False, method="Boosting", verbose=verbose
            )
        except TypeError:
            print(f"Skipped boosting ensemble. Estimator {tuned} not supported.")

        exp.stack_models(
            estimator_list=best_methods,
            choose_better=False,
            restack=False,
            verbose=verbose,
        )

        exp.blend_models(
            estimator_list=best_methods,
            choose_better=False,
            optimize=sort_metric,
            verbose=verbose,
        )

    best_model = exp.automl(optimize=sort_metric)

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


def persist_data(
    experiment: RegressionExperiment,
    folder: str = "artifacts/data",
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
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = f"{folder}/{experiment.exp_name_log}.parquet"
        data.to_parquet(path)
        return path

    raise ValueError("unknown saving strategy")


def persist_model(
    experiment: RegressionExperiment,
    model,
    folder: str = "artifacts/models",
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
        model_folder.mkdir(parents=True, exist_ok=True)
        path_model = model_folder / Path(experiment.exp_name_log)
        experiment.save_model(model=model, model_name=str(path_model))
        return path_model

    raise ValueError("unknown saving strategy")


def get_model_paths(folder: str = "artifacts/models", stategy: str = "local"):
    """Retrieves a list of model files from the specified folder and subfolders.

    Recursive `Path.glob`.

    Args:
        folder (str, optional): Path to the folder containing model files. Defaults
        to "models". Folder will be created if not existing.

        file_extension (str, optional): File extension for model files. Defaults to
        "pkl".

        strategy (str, optional): Retrieval strategy. Currently, only "local" strategy
        is supported. Other strategies like MLflow might be supported in the future.

    Returns:
        List[Path]: A list of Path objects representing the model files in the specified
        folder.

    Raises:
        NotADirectoryError: If the specified folder does not exist or is not a
        directory.

        ValueError: If an unsupported retrieval strategy is specified.
    """
    if not Path(folder).is_dir():
        raise NotADirectoryError(f"'{folder}' either not existing or not a folder.")

    if stategy == "local":
        return list(Path(folder).glob("**/*.pkl"))

    raise ValueError("unknown saving strategy")


def persist_experiment(
    experiment: RegressionExperiment,
    folder: str = "artifacts/experiments",
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
        exp_folder.mkdir(parents=True, exist_ok=True)
        path_exp = f"{exp_folder}/{experiment.exp_name_log}.exp"
        experiment.save_experiment(path_or_file=path_exp)
        return path_exp

    raise ValueError("unknown saving strategy")
