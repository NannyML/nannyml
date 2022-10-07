#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class Model(SQLModel, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    runs: List["Run"] = Relationship(back_populates="model")


class Run(SQLModel, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: Optional[int] = Field(default=None, foreign_key="model.id")
    model: Model = Relationship(back_populates="runs")
    # metrics: List["Metric"] = Relationship(back_populates="run")  # Could not get this to work (due to inheritance)
    execution_timestamp: datetime = Field(default=datetime.now())


class Metric(SQLModel):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: Optional[int] = Field(default=None, foreign_key="model.id")
    run_id: int = Field(default=None, foreign_key="run.id")
    # run: Run = Relationship(back_populates="metrics")  # Could not get this to work (due to inheritance)
    start_timestamp: datetime
    end_timestamp: datetime
    timestamp: datetime  # 'center' timestamp
    metric_name: str
    value: float
    alert: bool


class StatisticalFeatureDriftMetric(Metric, table=True):  # type: ignore
    __tablename__ = 'statistical_feature_drift_metrics'

    feature_name: str


class DistanceFeatureDriftMetric(Metric, table=True):  # type: ignore
    __tablename__ = 'statistical_distance_drift_metrics'

    feature_name: str
    upper_threshold: Optional[float]
    lower_threshold: Optional[float]


class DataReconstructionFeatureDriftMetric(Metric, table=True):  # type: ignore
    __tablename__ = 'data_reconstruction_feature_drift_metrics'

    upper_threshold: Optional[float]
    lower_threshold: Optional[float]


class StatisticalOutputDriftMetric(Metric, table=True):  # type: ignore
    __tablename__ = 'statistical_output_drift_metrics'

    output_name: str


class TargetDriftMetric(Metric, table=True):  # type: ignore
    __tablename__ = 'target_drift_metrics'

    target_name: str


class RealizedPerformanceMetric(Metric, table=True):  # type: ignore
    __tablename__ = 'realized_performance_metrics'

    upper_threshold: Optional[float]
    lower_threshold: Optional[float]


class CBPEPerformanceMetric(Metric, table=True):  # type: ignore
    __tablename__ = "cbpe_performance_metrics"

    upper_threshold: Optional[float]
    lower_threshold: Optional[float]


class DLEPerformanceMetric(Metric, table=True):  # type: ignore
    __tablename__ = "dle_performance_metrics"

    upper_threshold: Optional[float]
    lower_threshold: Optional[float]
