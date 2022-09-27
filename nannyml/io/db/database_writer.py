from typing import Any, Dict, Optional

from sqlmodel import Session, SQLModel, create_engine, select

from nannyml._typing import Result
from nannyml.exceptions import WriterException
from nannyml.io.base import Writer
from nannyml.io.db.entities import Model, Run
from nannyml.io.db.mappers import MapperFactory


class DatabaseWriter(Writer):
    def __init__(
        self,
        connection_string: str,
        connection_options: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ):
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
