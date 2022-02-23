#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for drift plotting module."""

import pytest

import nannyml as nml
from nannyml import DriftPlots, InvalidArgumentsException
from nannyml.drift import UnivariateStatisticalDriftCalculator
from nannyml.exceptions import CalculatorNotFittedException


@pytest.fixture
def sample_data():  # noqa: D103
    ref_data, ana_data, _ = nml.datasets.load_synthetic_sample()
    return ref_data, ana_data


@pytest.fixture
def sample_metadata(sample_data):  # noqa: D103
    ref_data, _ = sample_data
    md = nml.extract_metadata(ref_data)
    md.ground_truth_column_name = 'work_home_actual'
    return md


@pytest.fixture
def sample_univariate_statistical_calculator(
    sample_metadata, sample_data  # noqa: D103
) -> UnivariateStatisticalDriftCalculator:
    calc = UnivariateStatisticalDriftCalculator(model_metadata=sample_metadata, chunk_size=5000)
    ref_data, _ = sample_data
    calc.fit(ref_data)
    return calc


@pytest.fixture
def sample_univariate_statistical_drift_result(sample_univariate_statistical_calculator, sample_data):  # noqa: D103
    _, data = sample_data
    return sample_univariate_statistical_calculator.calculate(data)


def test_drift_plots_init_raises_calculator_not_fitted_when_calculator_has_no_chunker(sample_metadata):  # noqa: D103
    with pytest.raises(CalculatorNotFittedException, match="the chunker for the provided calculator is not set."):
        _ = DriftPlots(drift_calculator=UnivariateStatisticalDriftCalculator(sample_metadata, chunk_size=5000))


def test_univariate_statistical_drift_raises_invalid_args_when_feature_label_or_column_name_both_none(  # noqa: D103
    sample_univariate_statistical_calculator, sample_univariate_statistical_drift_result
):
    plots = DriftPlots(sample_univariate_statistical_calculator)
    with pytest.raises(
        InvalidArgumentsException, match="one of 'feature_label' or 'feature_column_name' should be provided."
    ):
        plots.plot_univariate_statistical_drift(sample_univariate_statistical_drift_result)


def test_univariate_statistical_drift_raises_invalid_args_when_feature_label_not_found_in_metadata(  # noqa: D103
    sample_univariate_statistical_calculator, sample_univariate_statistical_drift_result
):
    plots = DriftPlots(sample_univariate_statistical_calculator)
    with pytest.raises(InvalidArgumentsException, match="could not find a feature foo"):
        plots.plot_univariate_statistical_drift(sample_univariate_statistical_drift_result, feature_label='foo')


def test_univariate_statistical_drift_raises_invalid_args_when_feature_column_name_not_found_in_metadata(  # noqa: D103
    sample_univariate_statistical_calculator, sample_univariate_statistical_drift_result
):
    plots = DriftPlots(sample_univariate_statistical_calculator)
    with pytest.raises(InvalidArgumentsException, match="could not find a feature foo"):
        plots.plot_univariate_statistical_drift(sample_univariate_statistical_drift_result, feature_column_name='foo')
