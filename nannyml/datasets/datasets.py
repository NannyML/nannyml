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


def load_synthetic_binary_classification_dataset():
    """Loads the synthetic binary classification dataset provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic binary classification dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic binary classification dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic binary
        classification dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_binary_classification_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_synthetic_binary_classification_dataset()

    """
    reference = load_csv_file_to_df('synthetic_sample_reference.csv')
    analysis = load_csv_file_to_df('synthetic_sample_analysis.csv')
    analysis_gt = load_csv_file_to_df('synthetic_sample_analysis_gt.csv')

    return reference, analysis, analysis_gt


def load_synthetic_multiclass_classification_dataset():
    """Loads the synthetic multiclass classification dataset provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic multiclass classification dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic multiclass classification dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic
        multiclass classification dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_multiclass_classification_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_synthetic_multiclass_classification_dataset()

    """
    reference = load_csv_file_to_df('mc_reference.csv')
    analysis = load_csv_file_to_df('mc_analysis.csv')
    analysis_gt = load_csv_file_to_df('mc_analysis_gt.csv')

    return reference, analysis, analysis_gt


def load_modified_california_housing_dataset():
    """Loads the modified california housing dataset provided for testing the NannyML package.

    This dataset has been altered to represent a binary classification problem over time.
    More information about the dataset can be found at:
    :ref:`dataset-california`

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of modified california housing dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of modified california housing dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of modified california housing dataset

    Examples
    --------
    >>> from nannyml.datasets import load_modified_california_housing_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_modified_california_housing_dataset()
    """
    reference = load_csv_file_to_df('california_housing_reference.csv')
    analysis = load_csv_file_to_df('california_housing_analysis.csv')
    analysis_gt = load_csv_file_to_df('california_housing_analysis_gt.csv')

    return reference, analysis, analysis_gt


def load_synthetic_car_loan_dataset():
    """Loads the synthetic car loan binary classification dataset provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic binary classification dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic binary classification dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic binary
        classification dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_car_loan_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_synthetic_car_loan_dataset()

    """
    reference = load_csv_file_to_df('synthetic_car_loan_reference.csv')
    analysis = load_csv_file_to_df('synthetic_car_loan_analysis.csv')
    analysis_gt = load_csv_file_to_df('synthetic_car_loan_analysis_target.csv')

    return reference, analysis, analysis_gt


def load_synthetic_car_price_dataset():
    """Loads the synthetic car price dataset provided for testing the NannyML package on regression problems.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic car price dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic car price dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic car price dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_car_price_dataset
    >>> reference, analysis, analysis_tgt = load_synthetic_car_price_dataset()

    """

    reference = load_csv_file_to_df('regression_synthetic_reference.csv')
    analysis = load_csv_file_to_df('regression_synthetic_analysis.csv')
    analysis_tgt = load_csv_file_to_df('regression_synthetic_analysis_targets.csv')

    return reference, analysis, analysis_tgt
