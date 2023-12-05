"""Meant as sample script."""
# %%
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment

data = get_data("traffic")

exp = RegressionExperiment()
exp.setup(
    data=data,
    target="traffic_volume",
    verbose=False,
    log_experiment=True,
    log_data=True,
    log_plots=True,
    # log_profile=True,
)

exp.compare_models(exclude=["lightgbm"])
