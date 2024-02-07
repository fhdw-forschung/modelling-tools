"""Tracking Resources."""
import os

import mlflow
import pandas as pd

from fhdw.modelling.evaluation import get_regression_metrics


def log_metrics_to_mlflow(y_true: pd.Series, y_pred: pd.Series, prefix: str = ""):
    """Log metrics to active MLflow Experiment and Run.

    Args:
        y_true (``pandas.Series``): The ground truth values.

        y_pred (``pandas.Series``): The values made by model inference.

        prefix (``str``): Prefix for the metric names. This could be used to specify the
            metrics' purpose. Example: When set to *test_* the resulting metrics are
            e.g. *test_RMSE*.
    """
    prefix = f"{prefix}_" if prefix and prefix[-1] != "_" else prefix

    metrics = get_regression_metrics(y_true=y_true, y_pred=y_pred)

    # filter out none values, because it is not allowed to log None values to mlflow
    metrics = {key: value for key, value in metrics.items() if value is not None}

    metrics = {f"{prefix}{metric}": v for metric, v in metrics.items()}
    mlflow.log_metrics(metrics=metrics)
    return True


class ModelManagement:
    """Pycaret Registered Model Management Unit."""

    def __init__(self, model_name: str) -> None:
        """Initialize PyCaret Model management.

        Args:
            model_name (str): The name of the registered model.
        """
        self.name = model_name
        self.client = mlflow.tracking.MlflowClient()
        self.framework = self._get_framework(self.name)

    def _get_model_run(self, stage_name: str):
        model = self._get_model_info_at_stage(stage_name=stage_name)
        run = self.client.get_run(str(model.run_id))
        return run

    def _get_framework(self, stage_name: str):
        run = self._get_model_run(stage_name=stage_name)
        framework = run.data.tags["framework"]
        return framework

    def _get_parent_run_id(self, stage_name: str):
        run = self._get_model_run(stage_name=stage_name)
        parent_id = run.data.tags["mlflow.parentRunId"]
        return parent_id

    def _get_pycaret_artifact(self, dataset: str, stage_name: str):
        parent_id = self._get_parent_run_id(stage_name=stage_name)
        data = self.client.download_artifacts(run_id=parent_id, path=f"{dataset}.csv")
        return data

    def _get_tpot_artifact(self, name_ending: str, stage_name: str):
        run_id = self._get_model_run(stage_name=stage_name).info.run_id
        directory = self.client.download_artifacts(run_id=run_id, path="")
        files = os.listdir(directory)
        dataset = next((s for s in files if s.endswith(name_ending)), None)
        data = self.client.download_artifacts(run_id=run_id, path=f"{dataset}.csv")
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

    def get_train_data(
        self, stage_name: str = "Production", split_target: bool = False
    ):
        """Return the data the named registered model has been trained with."""
        if not split_target:
            if self.framework == "pycaret":
                data = self._get_pycaret_artifact(
                    dataset="Train", stage_name=stage_name
                )
                data = pd.read_csv(data)
            elif self.framework == "tpot":
                data_y = self._get_tpot_artifact(
                    name_ending="y_train", stage_name=stage_name
                )
                data_y = pd.read_csv(data_y)
                data = self._get_tpot_artifact(
                    name_ending="x_train", stage_name=stage_name
                )
                data = pd.read_csv(data).join(data_y)
            else:
                raise ValueError(
                    "Recieved unsupported framework. "
                    "The supported frameworks are pycaret and tpot."
                )
        else:
            if self.framework == "tpot":
                data_y = self._get_tpot_artifact(
                    name_ending="y_train", stage_name=stage_name
                )
                data_y = pd.read_csv(data_y)
                data = self._get_tpot_artifact(
                    name_ending="x_train", stage_name=stage_name
                )
                data = (data, data_y)
            else:
                raise ValueError(
                    "For split target = True the only supported framework is tpot."
                )

        return data

    def get_test_data(self, stage_name: str = "Production", split_target: bool = False):
        """Return the holdout set used during training of the registered model."""
        if not split_target:
            if self.framework == "pycaret":
                data = self._get_pycaret_artifact(dataset="Test", stage_name=stage_name)
                data = pd.read_csv(data)
            elif self.framework == "tpot":
                data_y = self._get_tpot_artifact(
                    name_ending="y_test", stage_name=stage_name
                )
                data_y = pd.read_csv(data_y)
                data = self._get_tpot_artifact(
                    name_ending="x_test", stage_name=stage_name
                )
                data = pd.read_csv(data).join(data_y)
            else:
                raise ValueError(
                    "Recieved unsupported framework. "
                    "The supported frameworks are pycaret and tpot."
                )
        else:
            if self.framework == "tpot":
                data_y = self._get_tpot_artifact(
                    name_ending="y_test", stage_name=stage_name
                )
                data_y = pd.read_csv(data_y)
                data = self._get_tpot_artifact(
                    name_ending="x_test", stage_name=stage_name
                )
                data = (data, data_y)
            else:
                raise ValueError(
                    "For split target = True the only supported framework is tpot."
                )
        return data

    def get_model_at_stage(self, stage_name: str = "Production"):
        """Get the trained model of the registered model from the given stage."""
        reg_path = f"models:/{self.name}/{stage_name}"
        model = mlflow.pyfunc.load_model(reg_path)
        return model
