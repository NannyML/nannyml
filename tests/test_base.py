
import numpy as np
import pandas as pd
import pytest

from nannyml.base import common_nan_removal


def test_common_nan_removal_dataframe():
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [9, 10, 11, np.nan]
    })
    selected_columns = ['A', 'B']
    df_cleaned, is_empty = common_nan_removal(data, selected_columns)

    expected_df = pd.DataFrame({
        'A': [1, 4],
        'B': [5, 8],
        'C': [9, np.nan]
    }).reset_index(drop=True)

    pd.testing.assert_frame_equal(df_cleaned, expected_df, check_dtype=False)  # ignore types because of infer_objects
    assert not is_empty


def test_common_nan_removal_dataframe_all_nan():
    data = pd.DataFrame({
        'A': [np.nan, np.nan],
        'B': [np.nan, np.nan],
        'C': [np.nan, np.nan]
    })
    selected_columns = ['A', 'B']
    df_cleaned, is_empty = common_nan_removal(data, selected_columns)

    expected_df = pd.DataFrame(columns=['A', 'B', 'C'])

    pd.testing.assert_frame_equal(df_cleaned, expected_df, check_index_type=False, check_dtype=False)
    assert is_empty


def test_common_nan_removal_ndarrays():
    data = [
        np.array([1, 5, 9]),
        np.array([2, np.nan, 10]),
        np.array([np.nan, 7, 11]),
        np.array([4, 8, np.nan])
    ]
    selected_columns_indices = [0, 1]  # Corresponds to columns 'A' and 'B'

    df_cleaned, is_empty = common_nan_removal(data, selected_columns_indices)

    expected_df = pd.DataFrame({
        'col_0': [1, 9],
        'col_1': [2, 10],
        'col_2': [np.nan, 11],
        'col_3': [4, np.nan],
    }).reset_index(drop=True)

    pd.testing.assert_frame_equal(df_cleaned, expected_df, check_dtype=False)
    assert not is_empty


def test_common_nan_removal_arrays_all_nan():
    data = [
        np.array([np.nan, np.nan]),
        np.array([np.nan, np.nan]),
        np.array([np.nan, np.nan]),

    ]
    selected_columns_indices = [0, 1]  # Corresponds to columns 'A' and 'B'

    df_cleaned, is_empty = common_nan_removal(data, selected_columns_indices)

    expected_df = pd.DataFrame(columns=[
        'col_0', 'col_1', 'col_2'
    ])

    pd.testing.assert_frame_equal(df_cleaned, expected_df, check_index_type=False, check_dtype=False)
    assert is_empty


def test_invalid_dataframe_columns():
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [9, 10, 11, np.nan]
    })
    selected_columns = ['A', 'D']  # 'D' does not exist
    with pytest.raises(ValueError):
        common_nan_removal(data, selected_columns)


def test_invalid_array_columns():
    data = [
        np.array([np.nan, np.nan]),
        np.array([np.nan, np.nan]),
        np.array([np.nan, np.nan]),

    ]
    selected_columns_indices = [0, 3]  # Index 3 does not exist in ndarray

    with pytest.raises(ValueError):
        common_nan_removal(data, selected_columns_indices)
