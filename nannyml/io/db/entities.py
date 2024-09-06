#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

""" Contains the definitions of the database entities that map directly to the underlying table definitions.

    Every ``Result`` class has a matching ``Entity`` class, which implies that each calculator or estimator will export
    its results into a specific table.
"""

import sys
from datetime import datetime
from typing import List, Optional

from pydantic import ConfigDict

try:
    from sqlmodel import Field, Relationship, SQLModel
except ImportError:
    raise ImportError(
        "`sqlmodel` module is not available. Please install the `nannyml[db]` extra to use this functionality."
    )


class Model(SQLModel, table=True):  # type: ignore[call-arg]
    """Represents a ``Model``.

    Only created when the ``model_name`` property of the :class:`~nannyml.io.db.database_writer.DatabaseWriter`
    was given.
    The ``id`` field here will act as a foreign key in the ``run`` table and all ``metric`` tables.

    Stored in the ``model`` table.
    """

    #: A technical key that is used as a foreign key in the other tables
    id: Optional[int] = Field(default=None, primary_key=True)

    #: Optional model name that might be useful in visualizations e.g. in Grafana dashboards
    name: str

    #: List of NannyML runs
    runs: List["Run"] = Relationship(back_populates="model")


class Run(SQLModel, table=True):  # type: ignore[call-arg]
    """Represents a NannyML run, allowing to filter results based on what run generated them.

    The ``id`` field here will act as a foreign key in all ``metric`` tables.

    Stored in the ``run`` table.
    """

    # Ignore clash of `model_id` field name with default protected namespace `model_`
    # See: https://github.com/pydantic/pydantic/discussions/7121
    # Better solution using `alias` is not possible due to SQLModel issue
    if sys.version_info >= (3, 8):
        model_config = ConfigDict(protected_namespaces=())  # type: ignore

    #: Foreign key in all ``metric`` tables
    id: Optional[int] = Field(default=None, primary_key=True)

    #: Used to link a run to a model
    model_id: Optional[int] = Field(default=None, foreign_key="model.id")

    #: The actual ``Model`` class instance that is linked to the run
    model: Model = Relationship(back_populates="runs")
    # metrics: List["Metric"] = Relationship(back_populates="run")  # Could not get this to work (due to inheritance)

    #: Execution time of NannyML run
    execution_timestamp: datetime = Field(default=datetime.now())


class Metric(SQLModel):
    """
    Base ``Metric`` definition.
    """

    # Ignore clash of `model_id` field name with default protected namespace `model_`
    # See: https://github.com/pydantic/pydantic/discussions/7121
    # Better solution using `alias` is not possible due to SQLModel issue
    if sys.version_info >= (3, 8):
        model_config = ConfigDict(protected_namespaces=())  # type: ignore

    #: The technical identifier for this database row
    id: Optional[int] = Field(default=None, primary_key=True)

    #: Foreign key pointing to a record in the ``model`` table
    model_id: Optional[int] = Field(default=None, foreign_key="model.id")

    #: Foreign key pointing to a record in the ``run`` table
    run_id: int = Field(default=None, foreign_key="run.id")

    # run: Run = Relationship(back_populates="metrics")  # Could not get this to work (due to inheritance)

    #: The start datetime of the :class:`~nannyml.chunk.Chunk`
    start_timestamp: datetime

    #: The end datetime of the :class:`~nannyml.chunk.Chunk`
    end_timestamp: datetime

    #: The ''center'' timestamp of the :class:`~nannyml.chunk.Chunk`, i.e. the mean of the start and end timestamps
    timestamp: datetime

    #: The name of the method being calculated, e.g. ``jensen_shannon`` or ``chi2``
    metric_name: str

    #: The value returned by the method
    value: float

    #: Indicates if the method raised an alert for this :class:`~nannyml.chunk.Chunk`
    alert: bool


class UnivariateDriftMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator`.

    Stored in the ``univariate_drift_metrics`` table.
    """

    __tablename__ = 'univariate_drift_metrics'

    #: The name of the column this metric belongs to
    column_name: str


class DataReconstructionFeatureDriftMetric(Metric, table=True):  # type: ignore[call-arg]
    """:class:`~nannyml.drift.multivariate.data_reconstruction.calculator.DataReconstructionDriftCalculator` results.

    Stored in the ``data_reconstruction_feature_drift_metrics`` table.
    """

    __tablename__ = 'data_reconstruction_feature_drift_metrics'

    #: The upper alerting threshold value
    upper_threshold: Optional[float]

    #: The lower alerting threshold value
    lower_threshold: Optional[float]


class RealizedPerformanceMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator`.

    Stored in the ``realized_performance_metrics`` table.
    """

    __tablename__ = 'realized_performance_metrics'

    #: The upper alerting threshold value
    upper_threshold: Optional[float]

    #: The lower alerting threshold value
    lower_threshold: Optional[float]


class CBPEPerformanceMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` estimator.

    Stored in the ``cbpe_performance_metrics`` table.
    """

    __tablename__ = "cbpe_performance_metrics"

    #: The upper alerting threshold value
    upper_threshold: Optional[float]

    #: The lower alerting threshold value
    lower_threshold: Optional[float]


class DLEPerformanceMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.performance_estimation.direct_loss_estimation.dle.DLE estimator`.

    Stored in the ``dle_performance_metrics`` table.
    """

    __tablename__ = "dle_performance_metrics"

    #: The upper alerting threshold value
    upper_threshold: Optional[float]

    #: The lower alerting threshold value
    lower_threshold: Optional[float]


class UnseenValuesMetric(Metric, table=True):
    __tablename__ = "unseen_values_metrics"

    #: The name of the column this metric belongs to
    column_name: str

    #: The upper alerting threshold value
    upper_threshold: Optional[float]

    #: The lower alerting threshold value
    lower_threshold: Optional[float]


class MissingValuesMetric(Metric, table=True):
    __tablename__ = "missing_values_metrics"

    #: The name of the column this metric belongs to
    column_name: str

    #: The upper alerting threshold value
    upper_threshold: Optional[float]

    #: The lower alerting threshold value
    lower_threshold: Optional[float]
