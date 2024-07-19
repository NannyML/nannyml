#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import functools
import importlib.util
import inspect
import logging
import os
import platform
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import segment.analytics as segment_analytics
from dotenv import load_dotenv

from nannyml import __version__
from nannyml._typing import ParamSpec

T = TypeVar('T')
P = ParamSpec('P')

# read any .env files to import environment variables
load_dotenv()


def disable_usage_logging():
    os.environ['NML_DISABLE_USAGE_LOGGING'] = '1'


def enable_usage_logging():
    if 'NML_DISABLE_USAGE_LOGGING' in os.environ:
        del os.environ['NML_DISABLE_USAGE_LOGGING']


class UsageEvent(str, Enum):
    """Logged usage events"""

    # Calculators

    STATS_COUNT_FIT = "Simple Stats Count fit"
    STATS_COUNT_RUN = "Simple Stats Count run"
    STATS_COUNT_PLOT = "Simple Stats Count plot"

    STATS_STD_FIT = "Simple Stats Std fit"
    STATS_STD_RUN = "Simple Stats Std run"
    STATS_STD_PLOT = "Simple Stats Std plot"

    STATS_AVG_FIT = "Simple Stats Avg fit"
    STATS_AVG_RUN = "Simple Stats Avg run"
    STATS_AVG_PLOT = "Simple Stats Avg plot"

    STATS_SUM_FIT = "Simple Stats Sum fit"
    STATS_SUM_RUN = "Simple Stats Sum run"
    STATS_SUM_PLOT = "Simple Stats Sum plot"

    STATS_MEDIAN_FIT = "Simple Stats Median fit"
    STATS_MEDIAN_RUN = "Simple Stats Median run"
    STATS_MEDIAN_PLOT = "Simple Stats Median plot"

    DQ_CALC_MISSING_VALUES_FIT = "Data Quality Calculator Missing Values fit"
    DQ_CALC_MISSING_VALUES_RUN = "Data Quality Calculator Missing Values run"
    DQ_CALC_MISSING_VALUES_PLOT = "Data Quality Calculator Missing Values plot"

    DQ_CALC_UNSEEN_VALUES_FIT = "Data Quality Calculator Unseen Values fit"
    DQ_CALC_UNSEEN_VALUES_RUN = "Data Quality Calculator Unseen Values run"
    DQ_CALC_UNSEEN_VALUES_PLOT = "Data Quality Calculator Unseen Values plot"

    DQ_CALC_VALUES_OUT_OF_RANGE_FIT = "Data Quality Calculator Values Out Of Range fit"
    DQ_CALC_VALUES_OUT_OF_RANGE_RUN = "Data Quality Calculator Values Out Of Range run"
    DQ_CALC_VALUES_OUT_OF_RANGE_PLOT = "Data Quality Calculator Values Out Of Range Plot"

    UNIVAR_DRIFT_CALC_FIT = "Univariate drift calculator fit"
    UNIVAR_DRIFT_CALC_RUN = "Univariate drift calculator run"
    UNIVAR_DRIFT_PLOT = "Univariate drift results plot"

    MULTIVAR_DRIFT_CALC_FIT = "Multivariate reconstruction error drift calculator fit"
    MULTIVAR_DRIFT_CALC_RUN = "Multivariate reconstruction error drift calculator run"
    MULTIVAR_DRIFT_PLOT = "Multivariate drift results plot"

    DC_CALC_FIT = "Domain Classifier calculator fit"
    DC_CALC_RUN = "Domain Classifier calculator run"
    DC_RESULTS_PLOT = "Domain Classifier results plot"

    PERFORMANCE_CALC_FIT = "Realized performance calculator fit"
    PERFORMANCE_CALC_RUN = "Realized performance calculator run"
    PERFORMANCE_PLOT = "Realized performance calculator plot"

    # Estimators

    CBPE_ESTIMATOR_FIT = "CBPE estimator fit"
    CBPE_ESTIMATOR_RUN = "CBPE estimator run"
    CBPE_PLOT = "CBPE estimator plot"

    DLE_ESTIMATOR_FIT = "DLE estimator fit"
    DLE_ESTIMATOR_RUN = "DLE estimator run"
    DLE_PLOT = "DLE estimator plot"

    # Ranking

    RANKER_ALERT_COUNT_RUN = "Run ranker using alert count"
    RANKER_CORRELATION_FIT = "Fit ranker using correlation with performance"
    RANKER_CORRELATION_RUN = "Run ranker using correlation with performance"

    CLI_RUN = "CLI run"

    WRITE_RAW = "Exported results with RawFilesWriter"
    WRITE_PICKLE = "Exported results with PickleWriter"
    WRITE_DB = "Exported results with DatabaseWriter"


class UsageLogger(ABC):
    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def log(self, usage_event: UsageEvent, metadata: Optional[Dict[str, Any]] = None):
        if "NML_DISABLE_USAGE_LOGGING" in os.environ:
            self._logger.debug(
                "found NML_DISABLE_USAGE_LOGGING key in environment variables. "
                f"Usage event {usage_event} not logged."
            )
            return
        if metadata is None:
            metadata = {}
        self._log(usage_event, metadata)
        self._logger.debug(f"logged usage for event {usage_event} with metadata {metadata}")

    @abstractmethod
    def _log(self, usage_event: UsageEvent, metadata: Dict[str, Any]):
        raise NotImplementedError(f"'{self.__class__.__name__}' does not implement '_log' yet.")


class SegmentUsageTracker(UsageLogger):
    SEGMENT_WRITE_KEY = 'lIVZJNAdj2ZaMzAHHnFWP76g7CuwmzGz'

    write_key: str

    def __init__(self, write_key: Optional[str] = None, machine_metadata: Optional[Dict[str, Any]] = None):
        if write_key is not None:
            self.write_key = write_key
        else:
            self.write_key = self.SEGMENT_WRITE_KEY
        segment_analytics.write_key = self.write_key

        segment_analytics.max_retries = 1

        segment_analytics.timeout = 3
        segment_analytics.max_retries = 1

        if machine_metadata is not None:
            self._identify(machine_metadata)

    def _identify(self, machine_metadata: Dict[str, Any]):
        segment_analytics.identify(machine_metadata)

    def _log(self, usage_event: UsageEvent, metadata: Dict[str, Any]):
        user_id = str(uuid.UUID(int=uuid.getnode()))
        metadata.update(_get_system_information())
        segment_analytics.track(user_id, usage_event.value, metadata)


DEFAULT_USAGE_LOGGER = SegmentUsageTracker()


def get_logger() -> UsageLogger:
    return DEFAULT_USAGE_LOGGER


def log_usage(
    usage_event: UsageEvent,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_from_self: Optional[List[str]] = None,
    metadata_from_kwargs: Optional[List[str]] = None,
    logger: UsageLogger = DEFAULT_USAGE_LOGGER,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def logging_decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def logging_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # track start times
            start_time = time.time()
            process_start_time = time.process_time()

            runtime_exception = None
            try:
                # run original function
                res = func(*args, **kwargs)
            except BaseException as exc:
                runtime_exception = exc

            # get run times
            run_time = time.time() - start_time
            process_run_time = time.process_time() - process_start_time

            try:
                # include run times in metadata
                md = metadata or {}
                md.update({'run_time': run_time, 'process_run_time': process_run_time})

                # report if an exception occurred
                md.update({'exception_occurred': False})
                if runtime_exception is not None:
                    if hasattr(runtime_exception, '__module__'):
                        exception_type = f'{runtime_exception.__module__}.{type(runtime_exception).__name__}'
                    else:
                        exception_type = type(runtime_exception).__name__
                    md.update(
                        {
                            'exception_occurred': True,
                            'exception_type': exception_type,
                        }
                    )

                # fetch additional information from instance properties
                if metadata_from_self is not None:
                    for attr in metadata_from_self:
                        val = getattr(args[0], attr)
                        if isinstance(val, List):
                            md.update({attr: [str(e) for e in val]})
                        else:
                            md.update({attr: str(val)})

                # fetch additional information from function kwargs
                if metadata_from_kwargs is not None:
                    for attr in metadata_from_kwargs:
                        if attr in kwargs:
                            md.update({attr: kwargs[attr]})
                        else:
                            # check if the requested parameter has a default value set
                            param = inspect.signature(func).parameters[attr]
                            if param.default is not inspect.Parameter.empty:
                                md.update({attr: param.default})

                # log the event
                logger.log(usage_event, md)
            finally:
                if runtime_exception is not None:
                    raise runtime_exception
                else:
                    return res

        return logging_wrapper

    return logging_decorator


def _get_system_information() -> Dict[str, Any]:
    return {
        "os_type": platform.system(),
        "runtime_environment": _get_runtime_environment(),
        "python_version": platform.python_version(),
        "nannyml_version": __version__,
        "nannyml_cloud": _is_nannyml_cloud(),
    }


def _is_nannyml_cloud():
    return 'NML_CLOUD' in os.environ


def _get_runtime_environment():
    if _is_running_in_notebook():
        return 'notebook'
    elif _is_running_in_docker():
        return 'docker'
    elif _is_running_in_kubernetes():
        if _is_running_in_aks():
            return 'aks'
        elif _is_running_in_eks():
            return 'eks'
        else:
            return 'kubernetes'
    else:
        return 'native'


# Inspired by https://github.com/jaraco/jaraco.docker/blob/main/jaraco/docker.py
def _is_running_in_docker():
    if Path('/.dockerenv').exists():
        return True

    if any('docker' in line for line in Path('/proc/self/cgroup').open()):
        return True

    return False


def _is_running_in_kubernetes():
    return Path('/var/run/secrets/kubernetes.io/').exists()


def _is_running_in_aks():
    import requests

    try:
        metadata = requests.get(
            'http://169.254.169.254/metadata/instance?api-version=2021-02-01', headers={'Metadata': 'true'}, timeout=5
        )
        return metadata.status_code == 200
    except Exception:
        return False


def _is_running_in_eks():
    import requests

    try:
        token = requests.put(
            'http://169.254.169.254/latest/api/token',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': 21600},
            timeout=5,
        ).raw()

        metadata = requests.get('http://169.254.169.254/latest/meta-data/', headers={'X-aws-ec2-metadata-token': token})
        return metadata.status_code == 200
    except Exception:
        return False


# Inspired by
# https://github.com/zenml-io/zenml/blob/275109da08b783d5d2cd508b5f703aed0c66e485/src/zenml/environment.py#L182
# and https://stackoverflow.com/a/39662359
def _is_running_in_notebook():
    if importlib.util.find_spec("IPython") is not None:
        from IPython import get_ipython

        if get_ipython().__class__.__name__ in [
            "ZMQInteractiveShell",
        ]:
            return True
    return False
