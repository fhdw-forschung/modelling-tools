"""Test evaluation resources."""


from fhdw.modelling.evaluation import plot_estimates_model_vs_actual


def test_plot_estimates_model_vs_actual():
    """Test vs-plot properties after generating the plot."""
    # Mock data for testing
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.2, 2.8, 3.7, 4.9]
    target_name = "Test Target"

    # Call the function to generate the plot
    figure = plot_estimates_model_vs_actual(y_true, y_pred, target_name)

    assert figure.layout.title.text == target_name  # type: ignore
    assert figure.layout.xaxis.title.text == "index"  # type: ignore
    assert figure.layout.yaxis.title.text == target_name  # type: ignore

    # Check if the data in the plot matches the input data
    assert figure.data[0].x.tolist() == list(range(len(y_true)))
    assert figure.data[0].y.tolist() == y_pred
    assert figure.data[2].y.tolist() == y_true
    assert len(figure.data) == 4
