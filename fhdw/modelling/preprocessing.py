"""Resources for preprocessing."""

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def remove_outliers_isolation(
    data: DataFrame,
    imputer_strategy: str = "median",
    random_state=None,
    verbose: bool = True,
) -> DataFrame:
    """Remove outliers with the ``IsolationForest`` algorithm.

    It is possible to pass a complete data set with independent (X) incl.
    dependent (y) variables

    Args:
        data: The DataFrame to be reduced.

        imputer_strategy: The Imputer strategy to be used.

        random_state: The random state to be used for the ``IsolationForest`` algorithm.

    Returns:
        The dataset reduced by the found outliers.
    """
    if verbose:
        print("Remove Outliers...")

    is_outlier = apply_isolation_forest(
        data=data,
        imputer_strategy=imputer_strategy,
        random_state=random_state,
        verbose=verbose,
    )

    # remove outliers from original dataset and return it (dummies not included)
    return data[~is_outlier]


def apply_isolation_forest(
    data: DataFrame,
    imputer_strategy: str = "median",
    random_state=None,
    verbose: bool = True,
):
    """Detect outliers with the ``IsolationForest`` algorithm.

    It is possible to pass a complete data set with independent (X) incl.
    dependent (y) variables

    For the application of the ``IsolationForest`` algorithm it is necessary that there
    are no ``NaN`` values in the data. Therefore an imputer is used. Here it uses
    ``median`` as the default strategy instead of simply replacing the ``NaN`` values
    with zeros ('0'), because there may be cases where there are only a few values in a
    column.
    In such a case, the zeros would form a data imbalance after replacement and would
    potentially be interpreted as 'normality' by the ``IsolationForest`` method.
    Therefore, it makes sense to replace with a customized strategy. Mean or median are
    suitable.

    Args:
        data: The DataFrame to be reduced.

        imputer_strategy: The Imputer strategy to be used.

        random_state: The random state to be used for the ``IsolationForest`` algorithm.

    Returns:
        The indices found to be anomal and therefore marked as outliers.

    Note:
        A ``sklearn.impute.SimpleImputer`` imputer is used. Read the respecting
        documentation on more detail about available strategies.
    """
    if verbose:
        print("Detect Outliers...")

    # non-numeric features are not allowed in IsolationForest; create temp dataset
    data_dummies = pd.get_dummies(data, drop_first=True)

    pipe = make_pipeline(
        SimpleImputer(strategy=imputer_strategy),
        IsolationForest(
            n_estimators=300, n_jobs=-2, verbose=verbose, random_state=random_state
        ),
    )

    # train and apply
    data_dummies["outlier"] = pipe.fit_predict(data_dummies)

    # determine and output outliers
    is_outlier = data_dummies["outlier"] == -1
    if verbose:
        print("number of anomalies:", sum(is_outlier))
        print(data_dummies[is_outlier].index)

    return is_outlier
