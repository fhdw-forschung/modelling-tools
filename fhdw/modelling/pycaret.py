"""Modelling resources utiliying pycaret for the process."""

from pandas import DataFrame
from pycaret.regression import RegressionExperiment

from fhdw.modelling.base import make_experiment_name


def create_regression_model(
    train_data: DataFrame,
    target: str,
    experiment_prefix: str = "",
    sort_metric: str = "RMSE",
    exclude_models: list | None = None,
):
    """Create a regression model with Pycaret.

    The model will be tuned and finalized. Models are saved to the

    Args:
        train_data: The training data.

        target: The name of the target variable in the train data.

        exp_name: The name to be assigned to the experiment.
        (For model logging and tracking.)

        sort_metric (str): The metric used to sort the models.

        exclude_models (List[str]): A list of model names to exclude from comparison.

    Returns:
        tuple: The RegressionExperiment and the trained Pipeline containing the model.
    """
    exp_name = make_experiment_name(target=target, prefix=experiment_prefix)
    print(f"experiment name: '{exp_name}'")

    # experiment setup
    exp = RegressionExperiment()
    exp.setup(data=train_data, target=target, experiment_name=exp_name)

    # model creation with picking best model and tuning, up to finalization
    best_method = exp.compare_models(exclude=exclude_models, sort=sort_metric)
    trained_model = exp.create_model(best_method)
    tuned_model = exp.tune_model(trained_model, choose_better=True)
    finalized_model = exp.finalize_model(tuned_model)

    # model persistence
    path_model = f"models/{experiment_prefix}{exp_name}"
    exp.save_model(model=finalized_model, model_name=path_model)
    print(f"saved model to '{path_model}'")

    return exp, finalized_model
