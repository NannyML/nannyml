#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

""" Contains the definitions of the database entities that map directly to the underlying table definitions.

    Every Result class has a matching Entity class, which implies that each calculator/estimator will export
    its results into a specific table.
"""

from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class Model(SQLModel, table=True):  # type: ignore
    """Represents a Model.

    Only created when the ``model_name`` property of the DatabaseWriter was given.
    The ``id`` field here will act as a foreign key in the ``run`` table and all ``metric`` tables.

    Stored in the ``model`` table.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    runs: List["Run"] = Relationship(back_populates="model")


class Run(SQLModel, table=True):  # type: ignore
    """Represents a NannyML run, allowing to filter results based on what run generated them.

    The ``id`` field here will act as a foreign key in all ``metric`` tables.

    Stored in the ``run`` table.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: Optional[int] = Field(default=None, foreign_key="model.id")
    model: Model = Relationship(back_populates="runs")
    # metrics: List["Metric"] = Relationship(back_populates="run")  # Could not get this to work (due to inheritance)
    execution_timestamp: datetime = Field(default=datetime.now())


class Metric(SQLModel):  # type: ignore
    """
    Base Metric definition
    """

    id: Optional[int] = Field(default=None, primary_key=True)  #: The technical identifier for this database row
    model_id: Optional[int] = Field(
        default=None, foreign_key="model.id"
    )  #: Foreign key pointing to a record in the ``model`` table
    run_id: int = Field(default=None, foreign_key="run.id")  #: Foreign key pointing to a record in the ``run`` table
    # run: Run = Relationship(back_populates="metrics")  # Could not get this to work (due to inheritance)
    start_timestamp: datetime  #: The start datetime of the Chunk
    end_timestamp: datetime  #: The end datetime of the Chunk
    timestamp: datetime  #: The 'center' timestamp of the Chunk, i.e. the mean of the start and end timestamps
    metric_name: str  #: The name of the method being calculated, e.g. 'jensen_shannon' or 'chi2'
    value: float  #: The value returned by the method
    alert: bool  #: Indicates if the method raised an alert for this Chunk


class UnivariateDriftMetric(Metric, table=True):  # type: ignore
    """Represents results of the UnivariateDriftCalculator.

    Stored in the ``univariate_drift_metrics`` table.
    """

    __tablename__ = 'univariate_drift_metrics'

    column_name: str  #: The name of the column this metric belongs to


class DataReconstructionFeatureDriftMetric(Metric, table=True):  # type: ignore
    """Represents results of the DataReconstructionDriftCalculator.

    Stored in the ``data_reconstruction_feature_drift_metrics`` table.
    """

    __tablename__ = 'data_reconstruction_feature_drift_metrics'

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value


class RealizedPerformanceMetric(Metric, table=True):  # type: ignore
    """Represents results of the RealizedPerformanceCalculator.

    Stored in the ``realized_performance_metrics`` table.
    """

    __tablename__ = 'realized_performance_metrics'

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value


class CBPEPerformanceMetric(Metric, table=True):  # type: ignore
    """Represents results of the CBPE estimator.

    Stored in the ``cbpe_performance_metrics`` table.
    """

    __tablename__ = "cbpe_performance_metrics"

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value


class DLEPerformanceMetric(Metric, table=True):  # type: ignore
    """Represents results of the DLE estimator.

    Stored in the ``dle_performance_metrics`` table.
    """

    __tablename__ = "dle_performance_metrics"

    upper_threshold: Optional[float]  #: The upper alerting threshold value
    lower_threshold: Optional[float]  #: The lower alerting threshold value
