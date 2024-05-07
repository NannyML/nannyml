import functools
import logging
import os
import platform
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from segment import analytics

from nannyml import __version__


class UsageEvent(str, Enum):
    """Tracked usage events"""

    # Calculators

    UNIVAR_STAT_DRIFT_CALC_FIT = "Univariate statistical drift calculator fit"
    UNIVAR_STAT_DRIFT_CALC_RUN = "Univariate statistical drift calculator run"
    MULTIVAR_RECONST_DRIFT_CALC_FIT = "Multivariate reconstruction error drift calculator fit"
    MULTIVAR_RECONST_DRIFT_CALC_RUN = "Multivariate reconstruction error drift calculator run"
    OUTPUT_DRIFT_CALC_FIT = "Output drift calculator fit"
    OUTPUT_DRIFT_CALC_RUN = "Output drift calculator run"
    TARGET_DISTRIBUTION_DRIFT_CALC_FIT = "Target distribution drift calculator fit"
    TARGET_DISTRIBUTION_DRIFT_CALC_RUN = "Target distribution drift calculator run"

    PERFORMANCE_CALC_FIT = "Realized performance calculator fit"
    PERFORMANCE_CALC_RUN = "Realized performance calculator run"

    # Estimators

    CBPE_ESTIMATOR_FIT = "CBPE estimator fit"
    CBPE_ESTIMATOR_RUN = "CBPE estimator run"


class UsageTracker(ABC):
    user_metadata: Dict[str, Any] = {}

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def track(self, usage_event: UsageEvent, metadata: Optional[Dict[str, Any]] = None):
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
    TEST_UUID = '58f6bfdf-fc5d-4166-aaa3-4949ceda8bdc'

    write_key: str

    def __init__(self, write_key: Optional[str] = None, user_metadata: Optional[Dict[str, Any]] = None):
        if write_key is not None:
            self.write_key = write_key
        else:
            self.write_key = self.SEGMENT_WRITE_KEY
        analytics.write_key = self.write_key

        analytics.max_retries = 1

        if user_metadata is not None:
            self._identify_user(user_metadata)

    def _identify_user(self, user_metadata: Dict[str, Any]):
        analytics.identify(user_metadata)

    def _track(self, usage_event: UsageEvent, metadata: Dict[str, Any]):
        user_id = str(uuid.UUID(int=uuid.getnode()))
        metadata.update(_get_system_information())
        analytics.track(user_id, usage_event.value, metadata)


DEFAULT_USAGE_TRACKER = SegmentUsageTracker()


def track(
    usage_event: UsageEvent, metadata: Optional[Dict[str, Any]] = None, tracker: UsageTracker = DEFAULT_USAGE_TRACKER
):
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

            # include run times in metadata
            md = metadata or {}
            md.update({'run_time': run_time, 'process_run_time': process_run_time})

            # track stuff
            tracker.track(usage_event, md)
            return res

        return tracking_wrapper

    return tracking_decorator


def _get_system_information() -> Dict[str, Any]:
    return {
        "os_architecture": platform.architecture(),
        "os_type": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "nannyml_version": __version__,
    }
