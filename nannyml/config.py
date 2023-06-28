#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2
import yaml
from pydantic import BaseModel, validator

from nannyml._typing import Self
from nannyml.exceptions import IOException
from nannyml.thresholds import Threshold

CONFIG_PATH_ENV_VAR_KEY = 'NML_CONFIG_PATH'


class InputDataConfig(BaseModel):
    path: str
    credentials: Optional[Dict[str, Any]]
    read_args: Optional[Dict[str, Any]]


class TargetDataConfig(InputDataConfig):
    join_column: Optional[str]


class InputConfig(BaseModel):
    reference_data: InputDataConfig
    analysis_data: InputDataConfig
    target_data: Optional[TargetDataConfig]


class WriterConfig(BaseModel):
    type: str
    params: Optional[Dict[str, Any]]
    write_args: Optional[Dict[str, Any]]


class ChunkerConfig(BaseModel):
    chunk_size: Optional[int]
    chunk_period: Optional[str]
    chunk_count: Optional[int]


class IntervalSchedulingConfig(BaseModel):
    weeks: Optional[int]
    days: Optional[int]
    hours: Optional[int]
    minutes: Optional[int]


class CronSchedulingConfig(BaseModel):
    crontab: str


class SchedulingConfig(BaseModel):
    interval: Optional[IntervalSchedulingConfig]
    cron: Optional[CronSchedulingConfig]


class StoreConfig(BaseModel):
    path: str
    credentials: Optional[Dict[str, Any]]
    filename: Optional[str]
    invalidate: bool = False


class CalculatorConfig(BaseModel):
    type: str
    name: Optional[str] = None
    enabled: Optional[bool] = True
    outputs: Optional[List[WriterConfig]]
    store: Optional[StoreConfig]
    params: Dict[str, Any]

    @validator('params')
    def _parse_thresholds(cls, value: Dict[str, Any]):
        """Parse thresholds in params and convert them to :class:`Threshold`'s"""
        # Some calculators expect `thresholds` parameter as dict
        thresholds = value.get('thresholds', {})
        for key, threshold in thresholds.items():
            thresholds[key] = Threshold.parse_object(threshold)

        # Multivariate calculator expects a single `threshold`
        threshold = value.get('threshold')
        if threshold is not None:
            value['threshold'] = Threshold.parse_object(threshold)

        return value


class Config(BaseModel):
    input: Optional[InputConfig]
    calculators: List[CalculatorConfig]
    scheduling: Optional[SchedulingConfig]

    ignore_errors: Optional[bool]

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, config_path: Optional[str] = None):
        with open(get_config_path(config_path), "r") as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
            return Config.parse_obj(config_dict)._render()

    @classmethod
    def parse(cls, config: str):
        config_dict = yaml.safe_load(config)
        return Config.parse_obj(config_dict)._render()

    def _render(self) -> Self:
        if self.input is not None:
            self.input.reference_data.path = _render_path_template(self.input.reference_data.path)
            self.input.analysis_data.path = _render_path_template(self.input.analysis_data.path)

            if self.input.target_data:
                self.input.target_data.path = _render_path_template(self.input.target_data.path)

        for config in self.calculators:
            for output in config.outputs or []:
                if output.params and 'path' in output.params:
                    output.params['path'] = _render_path_template(output.params['path'])

            if config.store:
                config.store.path = _render_path_template(config.store.path)

        return self


def get_config_path(custom_config_path: Optional[str] = None) -> Path:
    if custom_config_path:
        return Path(custom_config_path)

    if CONFIG_PATH_ENV_VAR_KEY in os.environ:
        return Path(os.environ[CONFIG_PATH_ENV_VAR_KEY])

    mounted_path = Path('/config/nannyml.yaml')
    if mounted_path.exists():
        return mounted_path

    cool_mounted_path = Path('/config/nann.yml')
    if cool_mounted_path.exists():
        return cool_mounted_path

    local_path = Path('nannyml.yaml')
    if local_path.exists():
        return local_path

    cool_path = Path('nann.yml')
    if cool_path.exists():
        return cool_path

    raise RuntimeError('could not determine config path')


def _render_path_template(path_template: str) -> str:
    try:
        env = jinja2.Environment()
        tpl = env.from_string(path_template)
        return tpl.render(
            minute=datetime.strftime(datetime.today(), "%M"),
            hour=datetime.strftime(datetime.today(), "%H"),
            day=datetime.strftime(datetime.today(), "%d"),
            weeknumber=date.today().isocalendar()[1],
            month=datetime.strftime(datetime.today(), "%m"),
            year=datetime.strftime(datetime.today(), "%Y"),
        )
    except Exception as exc:
        raise IOException(f"could not render file path template: '{path_template}': {exc}")
