#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

""" Contains the definitions of the database entities that map directly to the underlying table definitions.

    Every ``Result`` class has a matching ``Entity`` class, which implies that each calculator or estimator will export
    its results into a specific table.
"""

from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class Model(SQLModel, table=True):  # type: ignore[call-arg]
    """Represents a ``Model``.

    Only created when the ``model_name`` property of the :class:`~nannyml.io.db.database_writer.DatabaseWriter` was given.
    The ``id`` field here will act as a foreign key in the ``run`` table and all ``metric`` tables.

    Stored in the ``model`` table.
    """

    id: Optional[int] = Field(default=None, primary_key=True) #: A technical key that is used as a foreign key in the other tables
    name: str #: Optional model name that might be useful in visualizations e.g. in Grafana dashboards
    runs: List["Run"] = Relationship(back_populates="model") #: List of NannyML runs


class Run(SQLModel, table=True):  # type: ignore[call-arg]
    """Represents a NannyML run, allowing to filter results based on what run generated them.

    The ``id`` field here will act as a foreign key in all ``metric`` tables.

    Stored in the ``run`` table.
    """

    id: Optional[int] = Field(default=None, primary_key=True) #: Foreign key in all ``metric`` tables
    model_id: Optional[int] = Field(default=None, foreign_key="model.id") #: Used to link a run to a model
    model: Model = Relationship(back_populates="runs") #: The actual ``Model`` class instance that is linked to the run
    # metrics: List["Metric"] = Relationship(back_populates="run")  # Could not get this to work (due to inheritance)
    execution_timestamp: datetime = Field(default=datetime.now()) #: Execution time of NannyML run


class Metric(SQLModel):
    """
    Base ``Metric`` definition.
    """

    id: Optional[int] = Field(default=None, primary_key=True)  #: The technical identifier for this database row
    model_id: Optional[int] = Field(
        default=None, foreign_key="model.id"
    )  #: Foreign key pointing to a record in the ``model`` table
    run_id: int = Field(default=None, foreign_key="run.id")  #: Foreign key pointing to a record in the ``run`` table
    # run: Run = Relationship(back_populates="metrics")  # Could not get this to work (due to inheritance)
    start_timestamp: datetime  #: The start datetime of the :class:`~nannyml.chunk.Chunk`
    end_timestamp: datetime  #: The end datetime of the :class:`~nannyml.chunk.Chunk`
    timestamp: datetime  #: The ''center'' timestamp of the :class:`~nannyml.chunk.Chunk`, i.e. the mean of the start and end timestamps
    metric_name: str  #: The name of the method being calculated, e.g. ``jensen_shannon`` or ``chi2``
    value: float  #: The value returned by the method
    alert: bool  #: Indicates if the method raised an alert for this :class:`~nannyml.chunk.Chunk`


class UnivariateDriftMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator`.

    Stored in the ``univariate_drift_metrics`` table.
    """

    __tablename__ = 'univariate_drift_metrics'

    column_name: str  #: The name of the column this metric belongs to


class DataReconstructionFeatureDriftMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.drift.multivariate.data_reconstruction.calculator.DataReconstructionDriftCalculator`.

    Stored in the ``data_reconstruction_feature_drift_metrics`` table.
    """

    __tablename__ = 'data_reconstruction_feature_drift_metrics'

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value


class RealizedPerformanceMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator`.

    Stored in the ``realized_performance_metrics`` table.
    """

    __tablename__ = 'realized_performance_metrics'

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value


class CBPEPerformanceMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` estimator.

    Stored in the ``cbpe_performance_metrics`` table.
    """

    __tablename__ = "cbpe_performance_metrics"

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value


class DLEPerformanceMetric(Metric, table=True):  # type: ignore[call-arg]
    """Represents results of the :class:`~nannyml.performance_estimation.direct_loss_estimation.dle.DLE estimator`.

    Stored in the ``dle_performance_metrics`` table.
    """

    __tablename__ = "dle_performance_metrics"

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value
