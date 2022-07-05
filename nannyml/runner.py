# #  Author:   Niels Nuyttens  <niels@nannyml.com>
# #
# #  License: Apache Software License 2.0
#
# """Used as an access point to start using NannyML in its most simple form."""
# import logging
# import sys
# from typing import List
#
# import pandas as pd
#
# from nannyml._typing import Calculator, Estimator
# from nannyml.chunk import Chunker, DefaultChunker
# from nannyml.drift.model_inputs.multivariate.data_reconstruction import DataReconstructionDriftCalculator
# from nannyml.drift.model_inputs.univariate.statistical import UnivariateStatisticalDriftCalculator
# from nannyml.drift.model_outputs.univariate.statistical import (
#     UnivariateStatisticalDriftCalculator as OutputDriftCalculator,
# )
# from nannyml.drift.target.target_distribution import TargetDistributionCalculator
# from nannyml.io.base import Writer
# from nannyml.io.file_writer import FileWriter
# from nannyml.metadata import ModelMetadata
# from nannyml.metadata.extraction import extract_metadata
# from nannyml.performance_calculation import PerformanceCalculator
# from nannyml.performance_estimation.confidence_based import CBPE
#
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# def run(
#     reference_data: pd.DataFrame,
#     analysis_data: pd.DataFrame,
#     model_metadata: ModelMetadata = None,
#     chunker: Chunker = DefaultChunker(),
#     writer: Writer = FileWriter(filepath='out', data_format='parquet'),
# ):
#
#     # create metadata from reference data
#     if model_metadata is None:
#         model_metadata = extract_metadata(reference_data)
#
#     # create calculators / estimators
#     calculators: List[Calculator] = [
#         UnivariateStatisticalDriftCalculator(model_metadata, chunker=chunker),
#         DataReconstructionDriftCalculator(model_metadata, chunker=chunker),
#         OutputDriftCalculator(model_metadata, chunker=chunker),
#         TargetDistributionCalculator(model_metadata, chunker=chunker),
#         PerformanceCalculator(
#             model_metadata, chunker=chunker, metrics=['roc_auc', 'f1', 'precision',
#             'recall', 'specificity', 'accuracy']
#         ),
#     ]
#
#     estimators: List[Estimator] = [
#         CBPE(  # type: ignore
#             model_metadata=model_metadata,
#             chunker=chunker,
#             metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
#         )
#     ]
#
#     for calc in calculators:
#         logger.info(f'running {calc.__class__.__name__}')
#         calc.fit(reference_data)
#         results = calc.calculate(pd.concat([reference_data, analysis_data], ignore_index=True))
#         writer.write(results)
#
#     for estimator in estimators:
#         logger.info(f'running {estimator.__class__.__name__}')
#         estimator.fit(reference_data)
#         results = estimator.estimate(pd.concat([reference_data, analysis_data], ignore_index=True))
#         writer.write(results)
#
#     return
