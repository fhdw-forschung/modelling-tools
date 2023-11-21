"""General Modelling Resources."""


from pycaret.regression import RegressionExperiment


def create_regression_model(
    train_data,
    target,
    exp_name,
    path_model: str,
    sort_metric: str = "RMSE",
    exclude_models: list | None = None,
):
    """Create a regression model with Pycaret.

    The model will be tuned and finalized.

    Args:
        train_data: The training data.

        target: The name of the target variable in the train data.

        exp_name: The name to be assigned to the experiment.
        (For model logging and tracking.)

        path_model: The path, where the model should be saved to.

        sort_metric (str): The metric used to sort the models.

        exclude_models (List[str]): A list of model names to exclude from comparison.

    Returns:
        tuple: The RegressionExperiment and the trained Pipeline containing the model.
    """
    exp = RegressionExperiment()
    exp.setup(data=train_data, target=target, experiment_name=exp_name)

    best_method = exp.compare_models(exclude=exclude_models, sort=sort_metric)
    trained_model = exp.create_model(best_method)
    tuned_model = exp.tune_model(trained_model, choose_better=True)
    finalized_model = exp.finalize_model(tuned_model)

    exp.save_model(finalized_model, path_model)
    print(f"saved model to '{path_model}'")

    return exp, finalized_model
