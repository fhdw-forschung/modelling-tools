"""Preprocessing Tests."""

import numpy as np
import pandas as pd
import pytest

from fhdw.modelling.preprocessing import combine_and_update


def test_combine_and_update_valid():
    """Test case for valid combination and update."""
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [400, 500, 600]})
    new_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6]}, index=[1, 2, 3])

    result = combine_and_update(df1, new_df)

    expected_result = pd.DataFrame(
        {"A": [1.0, 1.0, 2.0, 3.0], "B": [400.0, 4.0, 600.0, 6.0]}
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_combine_and_update_columns_mismatch():
    """Test case for ValueError when columns differ."""
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [400, 500, 600]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "C": [4, np.nan, 6]})

    with pytest.raises(ValueError, match="columns of df1 and df2 must not differ"):
        combine_and_update(df1, df2)


def test_combine_and_update_index_names_mismatch():
    """Test case for ValueError when index names differ."""
    df1 = pd.DataFrame(
        {"A": [1, 2, 3], "B": [400, 500, 600]}, index=pd.Index([1, 2, 3], name="ID")
    )
    df2 = pd.DataFrame(
        {"A": [1, 2, 3], "B": [4, np.nan, 6]}, index=pd.Index([1, 2, 3], name="Index")
    )

    with pytest.raises(ValueError, match="indexes do not have the same name"):
        combine_and_update(df1, df2)
