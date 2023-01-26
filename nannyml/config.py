#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore
from pydantic import BaseModel

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


class RawFileWriterConfig(BaseModel):
    path: str
    format: str = 'parquet'
    credentials: Optional[Dict[str, Any]]
    write_args: Optional[Dict[str, Any]]


class DatabaseWriterConfig(BaseModel):
    connection_string: str
    model_name: Optional[str]


class PickleWriterConfig(BaseModel):
    path: str
    credentials: Optional[Dict[str, Any]]
    write_args: Optional[Dict[str, Any]]


class WriterConfig(BaseModel):
    database: Optional[DatabaseWriterConfig]
    raw_files: Optional[RawFileWriterConfig]
    pickle: Optional[PickleWriterConfig]


class ColumnMapping(BaseModel):
    features: List[str]
    timestamp: str
    y_pred: str
    y_pred_proba: Union[str, Dict[str, str], None]
    y_true: str


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


class FileStoreConfig(BaseModel):
    path: str


class StoreConfig(BaseModel):
    file: Optional[FileStoreConfig]


class Config(BaseModel):
    input: InputConfig
    output: WriterConfig
    column_mapping: ColumnMapping
    chunker: Optional[ChunkerConfig]
    scheduling: Optional[SchedulingConfig]
    store: Optional[StoreConfig]

    problem_type: str
    ignore_errors: Optional[bool]

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, config_path: Optional[str] = None):
        with open(get_config_path(config_path), "r") as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
            return Config.parse_obj(config_dict)


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
