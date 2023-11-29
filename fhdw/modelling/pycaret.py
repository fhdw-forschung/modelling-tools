"""Modelling resources utiliying pycaret for the process."""
from pathlib import Path

from pandas import DataFrame
from pycaret.regression import RegressionExperiment

from fhdw.modelling.base import make_experiment_name


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

    persist_model(experiment=exp, model=finalized_model, exp_name=exp_name)

    return exp, finalized_model


def persist_model(experiment, model, exp_name):
    """Persist the given model.

    Convenience function to manage `save_model` boilerplate.
    """
    model_folder = Path("models")
    model_folder.mkdir(exist_ok=True)
    path_model = f"{model_folder}/{exp_name}"
    experiment.save_model(model=model, model_name=path_model)
    print(f"saved model to '{path_model}'")
