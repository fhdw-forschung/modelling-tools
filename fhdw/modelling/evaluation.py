"""Collection of evaluation resources and methods."""

import pandas as pd
import plotly.express as px


def plot_estimates_model_vs_actual(y_true, y_pred, target: str):
    """Plot to compare estimates.

    Estimates made by the model with `experiment.predict_model` are plotted alongside
    with the actual values.

    Args:
        y_true: The actual values of the ground truth.

        y_pred: The inference values made by the model.

        target: The learning target. Will be used for titles and labels.
    """
    result = pd.DataFrame(
        {
            "Model": y_pred,
            "y_true": y_true,
        }
    )
    figure = px.scatter(
        result,
        x=result.index,
        y=["Model", "y_true"],
        title=target,
        labels={"value": target},
        hover_name=result.index,
        marginal_y="box",
    )
    return figure
