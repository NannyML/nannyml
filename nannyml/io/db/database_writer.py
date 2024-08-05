from typing import Any, Dict, Optional

try:
    from sqlmodel import Session, SQLModel, create_engine, select
except ImportError:
    raise ImportError(
        "`sqlmodel` module is not available. Please install the `nannyml[db]` extra to use this functionality."
    )

from nannyml._typing import Result
from nannyml.exceptions import WriterException
from nannyml.io.base import Writer, WriterFactory
from nannyml.io.db.entities import Model, Run
from nannyml.io.db.mappers import MapperFactory
from nannyml.usage_logging import UsageEvent, log_usage


@WriterFactory.register('database')  # registration name matches property used in configuration file
class DatabaseWriter(Writer):
    """A :class:`~nannyml.io.base.Writer` implementation that writes a ``Result`` into a database table.

    The ``Result`` class is transformed into a list of *DbMetric* objects
    by an appropriate :class:`~nannyml.io.db.mappers.Mapper` instance.
    These *DbMetrics* are written into a database table, specific to the ``Result`` class.

    This supports any database that is compatible with *SQLAlchemy*.
    """

    def __init__(
        self,
        connection_string: str,
        connection_options: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ):
        """
        Creates a new ``DatabaseWriter`` instance.

        Parameters
        ----------
        connection_string: str
            The connection string that configures the connection to the database.
            Might contain user credentials as well.
        connection_options: Dict[str, Any], default=None
            Additional options passed along to the underlying *SQLAlchemy* engine.
        model_name: str, default=None
            An optional name for the model being monitored. When given this will cause a record to be created in the
            ``models`` table and having each *DbMetric* link to that one. This allows easy filtering and dropdown
            population in data visualization tools in case of multiple models exporting into the same database
            structure.

        Examples
        --------
        >>> # write to local in-memory database
        >>> sqlite_writer = DatabaseWriter(connection_string='sqlite:///', model_name='car_loan_prediction')
        >>> sqlite_writer.write(result)

        >>> postgres_writer = DatabaseWriter(
        ...  connection_string='postgresql://postgres:mysecretpassword@localhost:5432/postgres',
        ...  model_name='car_loan_prediction'
        ... )
        >>> postgres_writer.write(result)
        """
        super().__init__()
        self.connection_string = connection_string
        if connection_options is None:
            connection_options = {}
        self._engine = create_engine(url=connection_string, **connection_options)
        try:
            SQLModel.metadata.create_all(self._engine)

            # find or create a 'model' and store the id
            self.model_id = self._upsert_model(model_name)

            # create the "run" and store the id
            self.run_id = self._create_run(model_id=self.model_id)
        except Exception as exc:
            raise WriterException(f"could not create DatabaseWriter: {exc}")

    @log_usage(UsageEvent.WRITE_DB)
    def _write(self, result: Result, **kwargs):
        mapper = MapperFactory.create(result)

        with Session(self._engine) as session:
            metrics = mapper.map_to_entity(result, run_id=self.run_id, model_id=self.model_id)
            session.add_all(metrics)
            session.commit()

    def _create_run(self, **run_args) -> int:
        """Inserts a new record into the 'run' table and returns the id."""
        run = Run(**run_args)
        with Session(self._engine) as session:
            session.add(run)
            session.commit()
            session.refresh(run)
            if run.id is None:
                raise RuntimeError("could not retrieve run identifier from the database")
            return run.id

    def _upsert_model(self, model_name: Optional[str] = None) -> Optional[int]:
        """Upsert a model given a model name, returns the model id."""

        # No model specified
        if model_name is None:
            return None

        with Session(self._engine) as session:
            model = session.exec(select(Model).where(Model.name == model_name)).first()
            if model is None:
                self._logger.info(f"could not find a model with name '{model_name}', creating new")
                model = Model(name=model_name)
                session.add(model)
                session.commit()

            return model.id
