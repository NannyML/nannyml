#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for drift plotting module."""

import pytest

from nannyml.datasets import load_synthetic_sample
from nannyml.drift import UnivariateStatisticalDriftCalculator
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import extract_metadata
from nannyml.plots import DriftPlots


@pytest.fixture
def sample_data():  # noqa: D103
    ref_data, ana_data, _ = load_synthetic_sample()
    return ref_data, ana_data


@pytest.fixture
def sample_metadata(sample_data):  # noqa: D103
    ref_data, _ = sample_data
    md = extract_metadata(ref_data)
    md.ground_truth_column_name = 'work_home_actual'
    return md


@pytest.fixture
def sample_univariate_statistical_calculator(  # noqa: D103
    sample_metadata, sample_data
) -> UnivariateStatisticalDriftCalculator:
    calc = UnivariateStatisticalDriftCalculator(model_metadata=sample_metadata, chunk_size=5000)
    ref_data, _ = sample_data
    calc.fit(ref_data)
    return calc


@pytest.fixture
def sample_univariate_statistical_drift_result(sample_univariate_statistical_calculator, sample_data):  # noqa: D103
    _, data = sample_data
    return sample_univariate_statistical_calculator.calculate(data)


def test_drift_plots_init_raises_invalid_arguments_exception_when_chunker_is_none(  # noqa: D103
    sample_univariate_statistical_calculator,
):
    metadata = sample_univariate_statistical_calculator.model_metadata
    with pytest.raises(InvalidArgumentsException, match="the provided chunker was 'None'"):
        _ = DriftPlots(model_metadata=metadata, chunker=None)


def test_univariate_statistical_drift_raises_invalid_args_when_feature_label_or_column_name_both_none(  # noqa: D103
    sample_univariate_statistical_calculator, sample_univariate_statistical_drift_result
):
    metadata = sample_univariate_statistical_calculator.model_metadata
    chunker = sample_univariate_statistical_calculator.chunker
    plots = DriftPlots(metadata, chunker)
    with pytest.raises(
        InvalidArgumentsException, match="one of 'feature_label' or 'feature_column_name' should be provided."
    ):
        plots.plot_univariate_statistical_drift(sample_univariate_statistical_drift_result)


def test_univariate_statistical_drift_raises_invalid_args_when_feature_label_not_found_in_metadata(  # noqa: D103
    sample_univariate_statistical_calculator, sample_univariate_statistical_drift_result
):
    metadata = sample_univariate_statistical_calculator.model_metadata
    chunker = sample_univariate_statistical_calculator.chunker
    plots = DriftPlots(metadata, chunker)
    with pytest.raises(InvalidArgumentsException, match="could not find a feature foo"):
        plots.plot_univariate_statistical_drift(sample_univariate_statistical_drift_result, feature_label='foo')


def test_univariate_statistical_drift_raises_invalid_args_when_feature_column_name_not_found_in_metadata(  # noqa: D103
    sample_univariate_statistical_calculator, sample_univariate_statistical_drift_result
):
    metadata = sample_univariate_statistical_calculator.model_metadata
    chunker = sample_univariate_statistical_calculator.chunker
    plots = DriftPlots(metadata, chunker)
    with pytest.raises(InvalidArgumentsException, match="could not find a feature foo"):
        plots.plot_univariate_statistical_drift(sample_univariate_statistical_drift_result, feature_column_name='foo')
