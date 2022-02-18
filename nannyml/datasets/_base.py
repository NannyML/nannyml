#  Author:  Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

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
    df: pandas DataFrame containing the requested data
    """
    with resources.path(DATA_MODULE, local_file) as data:
        return read_csv(data)


def load_synthetic_sample():
    """Loads the synthetic sample provided for testing the NannyML package.  # noqa: D202, D415

    Returns
    -------
    refererence : DataFrame
        dataframe containing reference partition of synthetic sample dataset
    analysis : DataFrame
        dataframe containing analysis partition of synthetic sample dataset
    analysis_gt : DataFrame
        dataframe containing ground truth results for the analysis partition of synthetic sample dataset
    """

    reference = load_csv_file_to_df('sythetic_sample_reference.csv')
    analysis = load_csv_file_to_df('sythetic_sample_analysis.csv')
    analysis_gt = load_csv_file_to_df('sythetic_sample_analysis_gt.csv')

    return reference, analysis, analysis_gt
