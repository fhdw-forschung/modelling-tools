"""Resources for preprocessing."""

import pandas as pd
from pandas import DataFrame


def combine_and_update(df1: DataFrame, df2: DataFrame):
    """Combine two pandas DataFrames and update values of first DataFrame.

    Updates the values of the first DataFrame with the non-null values from the second
    DataFrame.

    Make sure df1 and df2 have the same columns and index names.

    Args:
        df1 (DataFrame): The first DataFrame.
        df2 (DataFrame): The second DataFrame.

    Returns:
        DataFrame: A new DataFrame with combined values, where values from df2 have
        overwritten corresponding values in df1.

    Notes:
        - The function does not modify the original DataFrames (df1 and df2) in place.

        - The index of the resulting DataFrame is based on the union of the indices of
        df1 and df2.

        - Non-null values from df2 will overwrite corresponding values in df1.

    Example:
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [400, 500, 600]})
        ...     A   B
        ... 0   1   400
        ... 1   2   500
        ... 2   3   600

        >>> new_df = pd.DataFrame({"A": [1, 2, 3],
                                   "B": [4, np.nan, 6]}, index=[1, 2, 3])
        ...     A   B
        ... 1   1   4.0
        ... 2   2   NaN
        ... 3   3   6.0

        >>> combine_and_update(df1, df2) returns:
                A     B
            0   1.0   400.0
            1   1.0   4.0
            2   2.0   600.0
            3   3.0   6.0
    """
    if not df1.columns.equals(df2.columns):
        raise ValueError("columns of df1 and df2 must not differ.")
    if df1.index.name != df2.index.name:
        raise ValueError("indexes do not have the same name")

    combined = df1.reindex(pd.concat([df1, df2]).index.drop_duplicates())
    combined.update(df2, overwrite=True)
    return combined
