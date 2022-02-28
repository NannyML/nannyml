#  Author:  Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Utility module offering curated datasets for quick experimentation."""

from importlib import resources

from pandas import DataFrame, read_csv

DATA_MODULE = "nannyml.datasets.data"


def load_csv_file_to_df(local_file: str) -> DataFrame:
    """Loads a data file from within the NannyML package.

    Parameters
    ----------
    local_file : str, required
        string with the name of the data file to be loaded.

    Returns
    -------
    df: pd.DataFrame
        A DataFrame containing the requested data
    """
    with resources.path(DATA_MODULE, local_file) as data:
        return read_csv(data)


def load_synthetic_sample():
    """Loads the synthetic sample provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference partition of synthetic sample dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis partition of synthetic sample dataset
    analysis_gt : pd.DataFrame
        A DataFrame containing ground truth results for the analysis partition of synthetic sample dataset
    """
    reference = load_csv_file_to_df('synthetic_sample_reference.csv')
    analysis = load_csv_file_to_df('synthetic_sample_analysis.csv')
    analysis_gt = load_csv_file_to_df('synthetic_sample_analysis_gt.csv')

    return reference, analysis, analysis_gt


def load_modified_california_housing_dataset():
    """Loads the modified california housing dataset provided for testing the NannyML package.

    This dataset has been altered to represent a binary classification problem over time.
    More information about the dataset can be found at:
    :ref:`california-housing`

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference partition of modified california housing dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis partition of modified california housing dataset
    analysis_gt : pd.DataFrame
        A DataFrame containing ground truth results for the analysis partition of modified california housing dataset
    """
    reference = load_csv_file_to_df('california_housing_reference.csv')
    analysis = load_csv_file_to_df('california_housing_analysis.csv')
    analysis_gt = load_csv_file_to_df('california_housing_analysis_gt.csv')

    return reference, analysis, analysis_gt
