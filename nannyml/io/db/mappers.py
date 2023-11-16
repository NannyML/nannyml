#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

from nannyml.data_quality.missing.result import Result as MissingValuesResult
from nannyml.data_quality.unseen.result import Result as UnseenValuesResult
from nannyml.drift.multivariate.data_reconstruction.result import Result as DataReconstructionDriftResult
from nannyml.drift.univariate import Result as UnivariateDriftResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.io.db.entities import CBPEPerformanceMetric, DataReconstructionFeatureDriftMetric, DLEPerformanceMetric
from nannyml.io.db.entities import Metric
from nannyml.io.db.entities import Metric as DbMetric
from nannyml.io.db.entities import (
    MissingValuesMetric,
    RealizedPerformanceMetric,
    UnivariateDriftMetric,
    UnseenValuesMetric,
)
from nannyml.performance_calculation.result import Result as RealizedPerformanceResult
from nannyml.performance_estimation.confidence_based.results import Result as CBPEResult
from nannyml.performance_estimation.direct_loss_estimation.result import Result as DLEResult


class Mapper(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        """Maps a result to a list of :class:`~nannyml.io.db.entities.Metric` entities."""


def _fully_qualified_class_name(result):
    return f"{result.__module__}.{result.__name__}"


class MapperFactory:
    """A factory class that produces :class:`~nannyml.io.db.mappers.Mapper` instances for a given Result subclass."""

    registry: Dict[str, Type[Mapper]] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, result, kwargs: Optional[Dict[str, Any]] = None) -> Mapper:
        """Returns an instance for a given result class."""

        if kwargs is None:
            kwargs = {}

        key = _fully_qualified_class_name(result.__class__)

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown result class '{key}' given. "
                f"Currently registered result classes are: {list(cls.registry.keys())}"
            )

        mapper_class = cls.registry[key]
        return mapper_class(**kwargs)

    @classmethod
    def register(cls, result) -> Callable:
        key = _fully_qualified_class_name(result)

        def inner_wrapper(wrapped_class: Type[Mapper]) -> Type[Mapper]:
            if key in cls.registry:
                cls._logger().warning(f"re-registering Metric for result_class='{key}'")
            cls.registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper


@MapperFactory.register(UnivariateDriftResult)
class UnivariateDriftResultMapper(Mapper):
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        def _parse(
            column_name: str, metric_name: str, start_date: datetime, end_date: datetime, value, alert: bool
        ) -> DbMetric:
            timestamp = start_date + (end_date - start_date) / 2

            return UnivariateDriftMetric(
                column_name=column_name,
                metric_name=metric_name,
                start_timestamp=start_date,
                end_timestamp=end_date,
                timestamp=timestamp,
                value=value,
                alert=alert,
                **metric_args,
            )

        if not isinstance(result, UnivariateDriftResult):
            raise InvalidArgumentsException(f"{self.__class__.__name__} can not deal with '{type(result)}'")

        if result.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        res: List[Metric] = []

        for column_name in result.continuous_column_names:
            for method in result.continuous_method_names:
                res += (
                    result.filter(period='analysis')
                    .to_df()[
                        [
                            ('chunk', 'chunk', 'start_date'),
                            ('chunk', 'chunk', 'end_date'),
                            (column_name, method, 'value'),
                            (column_name, method, 'alert'),
                        ]
                    ]
                    .apply(lambda r: _parse(column_name, method, *r), axis=1)
                    .to_list()
                )

        for column_name in result.categorical_column_names:
            for method in result.categorical_method_names:
                res += (
                    result.filter(period='analysis')
                    .to_df()[
                        [
                            ('chunk', 'chunk', 'start_date'),
                            ('chunk', 'chunk', 'end_date'),
                            (column_name, method, 'value'),
                            (column_name, method, 'alert'),
                        ]
                    ]
                    .apply(lambda r: _parse(column_name, method, *r), axis=1)
                    .to_list()
                )

        return res


@MapperFactory.register(DataReconstructionDriftResult)
class ReconstructionErrorDriftResultMapper(Mapper):
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        def _parse(
            metric_name: str,
            start_date: datetime,
            end_date: datetime,
            value,
            upper_threshold,
            lower_threshold,
            alert,
        ) -> DbMetric:
            timestamp = start_date + (end_date - start_date) / 2

            return DataReconstructionFeatureDriftMetric(
                metric_name=metric_name,
                start_timestamp=start_date,
                end_timestamp=end_date,
                timestamp=timestamp,
                value=value,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                alert=alert,
                **metric_args,
            )

        if not isinstance(result, DataReconstructionDriftResult):
            raise InvalidArgumentsException(f"{self.__class__.__name__} can not deal with '{type(result)}'")

        if result.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        res: List[DbMetric] = []

        for metric in result.metrics:
            res += (
                result.filter(period='analysis')
                .to_df()[
                    [
                        ('chunk', 'start_date'),
                        ('chunk', 'end_date'),
                        (metric.column_name, 'value'),
                        (metric.column_name, 'upper_threshold'),
                        (metric.column_name, 'lower_threshold'),
                        (metric.column_name, 'alert'),
                    ]
                ]
                .apply(lambda r: _parse(metric.column_name, *r), axis=1)
                .to_list()
            )

        return res


@MapperFactory.register(RealizedPerformanceResult)
class RealizedPerformanceMapper(Mapper):
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        def _parse(
            metric_name: str,
            start_date: datetime,
            end_date: datetime,
            value,
            upper_threshold,
            lower_threshold,
            alert: bool,
        ) -> RealizedPerformanceMetric:
            timestamp = start_date + (end_date - start_date) / 2

            return RealizedPerformanceMetric(
                metric_name=metric_name,
                start_timestamp=start_date,
                end_timestamp=end_date,
                timestamp=timestamp,
                value=value,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                alert=alert,
                **metric_args,
            )

        if result.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        res: List[DbMetric] = []

        column_names = [column_name for metric in result.metrics for column_name in metric.column_names]

        for metric in column_names:
            res += (
                result.filter(partition='analysis')
                .to_df()[
                    [
                        ('chunk', 'start_date'),
                        ('chunk', 'end_date'),
                        (metric, 'value'),
                        (metric, 'upper_threshold'),
                        (metric, 'lower_threshold'),
                        (metric, 'alert'),
                    ]
                ]
                .apply(lambda r: _parse(metric, *r), axis=1)
                .to_list()
            )

        return res


@MapperFactory.register(CBPEResult)
class CBPEMapper(Mapper):
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        def _parse(
            metric_name: str,
            start_date: datetime,
            end_date: datetime,
            value,
            upper_threshold,
            lower_threshold,
            alert: bool,
        ) -> CBPEPerformanceMetric:
            timestamp = start_date + (end_date - start_date) / 2

            return CBPEPerformanceMetric(
                metric_name=metric_name,
                start_timestamp=start_date,
                end_timestamp=end_date,
                timestamp=timestamp,
                value=value,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                alert=alert,
                **metric_args,
            )

        if result.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        res: List[Metric] = []

        for metric in [column_name for metric in result.metrics for column_name in metric.column_names]:
            res += (
                result.filter(period='analysis')
                .to_df()[
                    [
                        ('chunk', 'start_date'),
                        ('chunk', 'end_date'),
                        (metric, 'value'),
                        (metric, 'upper_threshold'),
                        (metric, 'lower_threshold'),
                        (metric, 'alert'),
                    ]
                ]
                .apply(lambda r: _parse(metric, *r), axis=1)
                .to_list()
            )

        return res


@MapperFactory.register(DLEResult)
class DLEMapper(Mapper):
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        def _parse(
            metric_name: str,
            start_date: datetime,
            end_date: datetime,
            value,
            upper_threshold,
            lower_threshold,
            alert: bool,
        ) -> DLEPerformanceMetric:
            timestamp = start_date + (end_date - start_date) / 2

            return DLEPerformanceMetric(
                metric_name=metric_name,
                start_timestamp=start_date,
                end_timestamp=end_date,
                timestamp=timestamp,
                value=value,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                alert=alert,
                **metric_args,
            )

        if result.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        res: List[DbMetric] = []

        for metric in [metric.column_name for metric in result.metrics]:
            res += (
                result.filter(period='analysis')
                .to_df()[
                    [
                        ('chunk', 'start_date'),
                        ('chunk', 'end_date'),
                        (metric, 'value'),
                        (metric, 'upper_threshold'),
                        (metric, 'lower_threshold'),
                        (metric, 'alert'),
                    ]
                ]
                .apply(lambda r: _parse(metric, *r), axis=1)
                .to_list()
            )

        return res


@MapperFactory.register(UnseenValuesResult)
class UnseenValuesResultMapper:
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        def _parse(
            column_name: str,
            start_date: datetime,
            end_date: datetime,
            value,
            upper_threshold,
            lower_threshold,
            alert: bool,
        ) -> UnseenValuesMetric:
            timestamp = start_date + (end_date - start_date) / 2

            return UnseenValuesMetric(
                column_name=column_name,
                metric_name="count",
                start_timestamp=start_date,
                end_timestamp=end_date,
                timestamp=timestamp,
                value=value,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                alert=alert,
                **metric_args,
            )

        if result.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        columns: List[str] = list(
            filter(lambda col: col != 'chunk', result.to_df().columns.get_level_values(0).drop_duplicates())
        )

        res: List[DbMetric] = []

        for column in columns:
            res += (
                result.filter(period='analysis')
                .to_df()[
                    [
                        ('chunk', 'start_date'),
                        ('chunk', 'end_date'),
                        (column, 'value'),
                        (column, 'upper_threshold'),
                        (column, 'lower_threshold'),
                        (column, 'alert'),
                    ]
                ]
                .apply(lambda r: _parse(column, *r), axis=1)
                .to_list()
            )

        return res


@MapperFactory.register(MissingValuesResult)
class MissingValuesResultMapper:
    def map_to_entity(self, result, **metric_args) -> List[DbMetric]:
        def _parse(
            column_name: str,
            start_date: datetime,
            end_date: datetime,
            value,
            upper_threshold,
            lower_threshold,
            alert: bool,
        ) -> MissingValuesMetric:
            timestamp = start_date + (end_date - start_date) / 2

            return MissingValuesMetric(
                column_name=column_name,
                metric_name="count",
                start_timestamp=start_date,
                end_timestamp=end_date,
                timestamp=timestamp,
                value=value,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                alert=alert,
                **metric_args,
            )

        if result.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        columns: List[str] = list(
            filter(lambda col: col != 'chunk', result.to_df().columns.get_level_values(0).drop_duplicates())
        )

        res: List[DbMetric] = []

        for column in columns:
            res += (
                result.filter(period='analysis')
                .to_df()[
                    [
                        ('chunk', 'start_date'),
                        ('chunk', 'end_date'),
                        (column, 'value'),
                        (column, 'upper_threshold'),
                        (column, 'lower_threshold'),
                        (column, 'alert'),
                    ]
                ]
                .apply(lambda r: _parse(column, *r), axis=1)
                .to_list()
            )

        return res
