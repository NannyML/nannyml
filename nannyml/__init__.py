# Author:   Niels Nuyttens  <niels@nannyml.com>
#
# License: Apache Software License 2.0

# TODO wording

"""The NannyML library, helping you maintain model performance since 2020.

Use the library to:
- Calculate drift on model inputs, outputs and ground truth
- Calculate performance metrics for your model
- Estimate model performance metrics in the absence of ground truth
"""

__author__ = """Niels Nuyttens"""
__email__ = 'niels@nannyml.com'

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.13.1'

import logging

from dotenv import load_dotenv
from importlib import import_module


from .calibration import Calibrator, IsotonicCalibrator, needs_calibration
from .chunk import Chunk, Chunker, CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from .data_quality import MissingValuesCalculator, NumericalRangeCalculator, UnseenValuesCalculator
from .datasets import (
    load_modified_california_housing_dataset,
    load_synthetic_binary_classification_dataset,
    load_synthetic_car_loan_data_quality_dataset,
    load_synthetic_car_loan_dataset,
    load_synthetic_car_price_dataset,
    load_synthetic_multiclass_classification_dataset,
    load_titanic_dataset,
    load_us_census_ma_employment_data,
)
from .distribution import CategoricalDistributionCalculator, ContinuousDistributionCalculator
from .drift import (
    AlertCountRanker,
    CorrelationRanker,
    DataReconstructionDriftCalculator,
    DomainClassifierCalculator,
    UnivariateDriftCalculator,
)
from .exceptions import ChunkerException, InvalidArgumentsException, MissingMetadataException
from .io import PickleFileWriter, RawFilesWriter
from .performance_calculation import PerformanceCalculator
from .performance_estimation import CBPE, DLE
from .stats import (
    SummaryStatsAvgCalculator,
    SummaryStatsMedianCalculator,
    SummaryStatsRowCountCalculator,
    SummaryStatsStdCalculator,
    SummaryStatsSumCalculator,
)
from .usage_logging import UsageEvent, disable_usage_logging, enable_usage_logging, log_usage


_optional_dependencies = {
    'DatabaseWriter': '.io.db',
}


def __getattr__(name: str):
    optional_module_path = _optional_dependencies.get(name)
    if optional_module_path is not None:
        module = import_module(optional_module_path, package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


try:
    import nannyml_premium

    logging.getLogger().debug('loaded "nannyml_premium" package')
except Exception:
    pass

# read any .env files to import environment variables
load_dotenv()

logging.getLogger(__name__).addHandler(logging.NullHandler())
