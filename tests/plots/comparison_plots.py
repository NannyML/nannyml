#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import pytest
from pytest_lazyfixture import lazy_fixture

from nannyml.chunk import Chunker, SizeBasedChunker
from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.multivariate.data_reconstruction import Result as DataReconstructionResult
from nannyml.drift.univariate import Result as UnivariateDriftResult
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_calculation import Result as RealizedPerformanceResult
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based import Result as CBPEResult


@pytest.fixture(scope='module')
def chunker() -> Chunker:
    return SizeBasedChunker(chunk_size=5000)


@pytest.fixture(scope='module')
def data_reconstruction_results(chunker) -> DataReconstructionResult:
    ref, ana, _ = load_synthetic_binary_classification_dataset()
    column_names = [col for col in ref.columns if col not in ['identifier', 'work_home_actual', 'timestamp']]
    calc = DataReconstructionDriftCalculator(column_names=column_names, chunker=chunker).fit(ref)
    result = calc.calculate(ana)
    assert isinstance(result, DataReconstructionResult)
    return result


@pytest.fixture(scope='module')
def univariate_drift_results(chunker) -> UnivariateDriftResult:
    ref, ana, _ = load_synthetic_binary_classification_dataset()
    column_names = [col for col in ref.columns if col not in ['identifier', 'work_home_actual', 'timestamp']]
    calc = UnivariateDriftCalculator(column_names=column_names, chunker=chunker).fit(ref)
    result = calc.calculate(ana)
    assert isinstance(result, UnivariateDriftResult)
    return result


@pytest.fixture(scope='module')
def realized_performance_results(chunker) -> RealizedPerformanceResult:
    ref, ana, tgt = load_synthetic_binary_classification_dataset()
    calc = PerformanceCalculator(
        metrics=['f1'],
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='work_home_actual',
        problem_type='classification',
        chunker=chunker,
    ).fit(ref)
    result = calc.calculate(ana.merge(tgt), on='identifier')
    assert isinstance(result, RealizedPerformanceResult)
    return result


@pytest.fixture(scope='module')
def cbpe_results(chunker) -> CBPEResult:
    ref, ana, tgt = load_synthetic_binary_classification_dataset()
    est = CBPE(
        metrics=['f1'],
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='work_home_actual',
        problem_type='classification',
        chunker=chunker,
    ).fit(ref)
    result = est.estimate(ana)
    assert isinstance(result, CBPEResult)
    return result


@pytest.mark.parametrize(
    'result_1, result_2',
    [
        (lazy_fixture('univariate_drift_results'), lazy_fixture('data_reconstruction_results')),
        (lazy_fixture('univariate_drift_results'), lazy_fixture('realized_performance_results')),
        (lazy_fixture('univariate_drift_results'), lazy_fixture('cbpe_results')),
        (lazy_fixture('data_reconstruction_results'), lazy_fixture('univariate_drift_results')),
        (lazy_fixture('data_reconstruction_results'), lazy_fixture('realized_performance_results')),
        (lazy_fixture('data_reconstruction_results'), lazy_fixture('cbpe_results')),
        (lazy_fixture('realized_performance_results'), lazy_fixture('univariate_drift_results')),
        (lazy_fixture('realized_performance_results'), lazy_fixture('data_reconstruction_results')),
        (lazy_fixture('realized_performance_results'), lazy_fixture('cbpe_results')),
        (lazy_fixture('cbpe_results'), lazy_fixture('univariate_drift_results')),
        (lazy_fixture('cbpe_results'), lazy_fixture('data_reconstruction_results')),
        (lazy_fixture('cbpe_results'), lazy_fixture('realized_performance_results')),
    ],
)
def test_comparison_plot_raises_no_exceptions(result_1, result_2):
    try:
        _ = result_1.compare(result_2).plot().show()
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_comparison_plot_comparing_multiple_metrics_raises_invalid_arguments_exception(chunker, cbpe_results):
    ref, ana, tgt = load_synthetic_binary_classification_dataset()
    calc = PerformanceCalculator(
        metrics=['f1', 'roc_auc'],
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='work_home_actual',
        problem_type='classification',
        chunker=chunker,
    ).fit(ref)
    drift_results = calc.calculate(ana.merge(tgt), on='identifier')

    with pytest.raises(
        InvalidArgumentsException,
        match="you're comparing 2 metrics to 2 metrics, but should only compare 1 to 1 at a time. "
        "Please filter yourresults first using `result.filter()",
    ):
        drift_results.compare(cbpe_results).plot().show()
