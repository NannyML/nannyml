import functools
import logging
import os
import platform
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import analytics as segment_analytics

from nannyml import __version__


def disable_user_analytics():
    os.environ['NML_DISABLE_USER_ANALYTICS'] = '1'


def enable_user_analytics():
    if 'NML_DISABLE_USER_ANALYTICS' in os.environ:
        del os.environ['NML_DISABLE_USER_ANALYTICS']


class UsageEvent(str, Enum):
    """Tracked usage events"""

    # Calculators

    UNIVAR_DRIFT_CALC_FIT = "Univariate drift calculator fit"
    UNIVAR_DRIFT_CALC_RUN = "Univariate drift calculator run"

    MULTIVAR_DRIFT_CALC_FIT = "Multivariate error drift calculator fit"
    MULTIVAR_DRIFT_CALC_RUN = "Multivariate error drift calculator run"

    PERFORMANCE_CALC_FIT = "Realized performance calculator fit"
    PERFORMANCE_CALC_RUN = "Realized performance calculator run"

    # Estimators

    CBPE_ESTIMATOR_FIT = "CBPE estimator fit"
    CBPE_ESTIMATOR_RUN = "CBPE estimator run"

    DLE_ESTIMATOR_FIT = "DLE estimator fit"
    DLE_ESTIMATOR_RUN = "DLE estimator run"

    CLI_RUN = "CLI run"


class UsageTracker(ABC):
    user_metadata: Dict[str, Any] = {}

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def track(self, usage_event: UsageEvent, metadata: Dict[str, Any] = None):
        if "NML_DISABLE_USER_ANALYTICS" in os.environ:
            self._logger.debug(
                "found NML_DISABLE_USER_ANALYTICS key in environment variables. "
                f"Usage event {usage_event} not tracked."
            )
            return
        if metadata is None:
            metadata = {}
        self._track(usage_event, metadata)
        self._logger.debug(f"tracked usage for event {usage_event} with metadata {metadata}")

    @abstractmethod
    def _track(self, usage_event: UsageEvent, metadata: Dict[str, Any]):
        raise NotImplementedError(f"'{self.__class__.__name__}' does not implement '_track' yet.")


class SegmentUsageTracker(UsageTracker):
    SEGMENT_WRITE_KEY = 'lIVZJNAdj2ZaMzAHHnFWP76g7CuwmzGz'

    write_key: str

    def __init__(self, write_key: str = None, user_metadata: Dict[str, Any] = None):
        if write_key is not None:
            self.write_key = write_key
        else:
            self.write_key = self.SEGMENT_WRITE_KEY
        segment_analytics.write_key = self.write_key

        segment_analytics.max_retries = 1

        if user_metadata is not None:
            self._identify_user(user_metadata)

    def _identify_user(self, user_metadata: Dict[str, Any]):
        segment_analytics.identify(user_metadata)

    def _track(self, usage_event: UsageEvent, metadata: Dict[str, Any]):
        user_id = str(uuid.UUID(int=uuid.getnode()))
        metadata.update(_get_system_information())
        segment_analytics.track(user_id, usage_event.value, metadata)


DEFAULT_USAGE_TRACKER = SegmentUsageTracker()


def track(usage_event: UsageEvent, metadata: Dict[str, Any] = None, tracker: UsageTracker = DEFAULT_USAGE_TRACKER):
    def tracking_decorator(func):
        @functools.wraps(func)
        def tracking_wrapper(*args, **kwargs):
            # track start times
            start_time = time.time()
            process_start_time = time.process_time()

            # run original function
            res = func(*args, **kwargs)

            # get run times
            run_time = time.time() - start_time
            process_run_time = time.process_time() - process_start_time

            try:
                # include run times in metadata
                md = metadata or {}
                md.update({'run_time': run_time, 'process_run_time': process_run_time})

                # track stuff
                tracker.track(usage_event, md)
            finally:
                return res

        return tracking_wrapper

    return tracking_decorator


def _get_system_information() -> Dict[str, Any]:
    return {
        "os_architecture": platform.architecture(),
        "os_type": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "runtime_environment": _get_runtime_environment(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "nannyml_version": __version__,
    }


def _get_runtime_environment():
    if _is_running_in_docker():
        return 'docker'
    else:
        return 'native'


# Inspired by https://github.com/jaraco/jaraco.docker/blob/main/jaraco/docker.py
def _is_running_in_docker():
    if Path('/.dockerenv').exists():
        return True

    if any('docker' in line for line in Path('/proc/self/cgroup').open()):
        return True

    return False
