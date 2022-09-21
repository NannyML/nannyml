from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

from nannyml._typing import Result
from nannyml.exceptions import WriterException
from nannyml.io.base import Writer


class Model(SQLModel, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    # runs: List["Run"] = Relationship(back_populates="model")


class Run(SQLModel, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    # model_id: int = Field(default=None, foreign_key="model.id")
    # model: Model = Relationship(back_populates="runs")
    metrics: List["Metric"] = Relationship(back_populates="run")
    execution_timestamp: datetime = Field(default=datetime.now())


class Metric(SQLModel, table=True):  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(default=None, foreign_key="run.id")
    run: Run = Relationship(back_populates="metrics")
    timestamp: datetime = Field(default=None)
    name: str
    value: float


class DatabaseWriter(Writer):
    def __init__(self, connection_string: str, **connection_opts):
        super().__init__()
        self.connection_string = connection_string
        self._engine = create_engine(url=connection_string, **connection_opts)
        try:
            SQLModel.metadata.create_all(self._engine)
        except Exception as exc:
            raise WriterException(f"could not create DatabaseWriter: {exc}")

    def _write(self, result: Result, **kwargs):

        run = Run()
        metrics = [
            Metric(name=metric.name, value=metric.value, run=run)
            for metric in result.to_metric_list()
        ]

        with Session(self._engine) as session:
            session.add(run)
            session.add_all(metrics)
            session.commit()
