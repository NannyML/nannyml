#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import os
from pathlib import Path
from typing import Dict

import yaml  # type: ignore

CONFIG_PATH_ENV_VAR_KEY = 'NML_CONFIG_PATH'


class Config:
    instance = None

    def __init__(self, config_dict: Dict):
        super().__init__()
        self.data = config_dict

    def __getitem__(self, item):
        return self.instance.data[item]

    @classmethod
    def load(cls, config_path: str = None):
        if not cls.instance:
            with open(get_config_path(config_path), "r") as config_file:
                config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
            cls.instance = cls(config_dict)
        return cls.instance


def get_config_path(custom_config_path: str = None) -> Path:
    if custom_config_path:
        return Path(custom_config_path)

    if CONFIG_PATH_ENV_VAR_KEY in os.environ:
        return Path(os.environ[CONFIG_PATH_ENV_VAR_KEY])

    mounted_path = Path('/config/config.yaml')
    if mounted_path.exists():
        return mounted_path

    local_path = Path('../config/config.yaml')
    if local_path.exists():
        return local_path

    raise RuntimeError('could not determine config path')
