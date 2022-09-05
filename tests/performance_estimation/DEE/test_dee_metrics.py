#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import pandas as pd
import pytest

from nannyml._typing import ProblemType
from nannyml.base import AbstractEstimator, AbstractEstimatorResult
from nannyml.performance_estimation.direct_error_estimation.metrics import MetricFactory


class FakeEstimator(AbstractEstimator):
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractEstimator:
        pass

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> AbstractEstimatorResult:
        pass


@pytest.mark.parametrize('metric', ['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'])
def test_metric_creation_with_non_dee_estimator_raises_runtime_exc(metric):
    with pytest.raises(RuntimeError, match='not an instance of type DEE'):
        MetricFactory.create(key=metric, problem_type=ProblemType.REGRESSION, kwargs={'estimator': FakeEstimator()})
