#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import List, Optional

import pandas as pd
import plotly.graph_objects
import pytest

from nannyml._typing import Key, Result, Self
from nannyml.base import Abstract1DResult, AbstractEstimator
from nannyml.chunk import DefaultChunker
from nannyml.datasets import load_synthetic_car_price_dataset
from nannyml.performance_estimation.direct_loss_estimation import DLE
from nannyml.performance_estimation.direct_loss_estimation.metrics import MAE, MAPE, MSE, MSLE, RMSE, RMSLE
from nannyml.thresholds import ConstantThreshold


class FakeEstimatorResult(Abstract1DResult):
    def keys(self) -> List[Key]:
        return []

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Self:
        return self

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        return plotly.graph_objects.Figure()


class FakeEstimator(AbstractEstimator):
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        return FakeEstimatorResult(results_data=pd.DataFrame())


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 20000},
            pd.DataFrame(
                {
                    'key': ['[0:19999]', '[20000:39999]', '[40000:59999]'],
                    'estimated_mae': [846.3459170892721, 781.3133425254717, 710.6939293960534],
                    'estimated_mape': [0.23258501209464247, 0.24455022221687936, 0.25789299683995814],
                    'estimated_mse': [1122386.0419119087, 994508.0216047806, 857904.9108674703],
                    'estimated_rmse': [1059.4272235089622, 997.2502301853735, 926.2315643873676],
                    'estimated_msle': [0.07135768434762862, 0.08298717136011495, 0.09557469460315275],
                    'estimated_rmsle': [0.26712859140801204, 0.2880749405278338, 0.30915157221523676],
                }
            ),
        ),
        (
            {'chunk_size': 20000, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:19999]', '[20000:39999]', '[40000:59999]'],
                    'estimated_mae': [846.3459170892721, 781.3133425254717, 710.6939293960534],
                    'estimated_mape': [0.23258501209464247, 0.24455022221687936, 0.25789299683995814],
                    'estimated_mse': [1122386.0419119087, 994508.0216047806, 857904.9108674703],
                    'estimated_rmse': [1059.4272235089622, 997.2502301853735, 926.2315643873676],
                    'estimated_msle': [0.07135768434762862, 0.08298717136011495, 0.09557469460315275],
                    'estimated_rmsle': [0.26712859140801204, 0.2880749405278338, 0.30915157221523676],
                }
            ),
        ),
        (
            {'chunk_number': 4},
            pd.DataFrame(
                {
                    'key': ['[0:14999]', '[15000:29999]', '[30000:44999]', '[45000:59999]'],
                    'estimated_mae': [848.017404062168, 845.9708028387239, 710.8740197712294, 712.9420253422753],
                    'estimated_mape': [0.232061608984753, 0.23269474575754298, 0.2580512295630102, 0.2572300572300006],
                    'estimated_mse': [1127872.106256183, 1119712.4738426104, 856837.1597713485, 861976.8926420709],
                    'estimated_rmse': [1062.0132326182113, 1058.164672365606, 925.654989599985, 928.4271068005667],
                    'estimated_msle': [0.07102886408763, 0.07146572703718296, 0.09569280211543528, 0.09503867384094684],
                    'estimated_rmsle': [0.2665124088811439, 0.2673307446538519, 0.3093425320182068, 0.3082834310191627],
                }
            ),
        ),
        (
            {'chunk_number': 4, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:14999]', '[15000:29999]', '[30000:44999]', '[45000:59999]'],
                    'estimated_mae': [848.017404062168, 845.9708028387239, 710.8740197712294, 712.9420253422753],
                    'estimated_mape': [0.232061608984753, 0.23269474575754298, 0.2580512295630102, 0.2572300572300006],
                    'estimated_mse': [1127872.106256183, 1119712.4738426104, 856837.1597713485, 861976.8926420709],
                    'estimated_rmse': [1062.0132326182113, 1058.164672365606, 925.654989599985, 928.4271068005667],
                    'estimated_msle': [0.07102886408763, 0.07146572703718296, 0.09569280211543528, 0.09503867384094684],
                    'estimated_rmsle': [0.2665124088811439, 0.2673307446538519, 0.3093425320182068, 0.3082834310191627],
                }
            ),
        ),
        (
            {'chunk_period': 'M', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2017-02', '2017-03'],
                    'estimated_mae': [840.3501574985254, 711.166997884045],
                    'estimated_mape': [0.23357839569346742, 0.2578266149411802],
                    'estimated_mse': [1110981.0256317134, 857741.4268967088],
                    'estimated_rmse': [1054.0308466224853, 926.1433079695112],
                    'estimated_msle': [0.07244657617776012, 0.09548339553615925],
                    'estimated_rmsle': [0.26915901652695967, 0.3090038762477896],
                }
            ),
        ),
        (
            {},
            pd.DataFrame(
                {
                    'key': [
                        '[0:5999]',
                        '[6000:11999]',
                        '[12000:17999]',
                        '[18000:23999]',
                        '[24000:29999]',
                        '[30000:35999]',
                        '[36000:41999]',
                        '[42000:47999]',
                        '[48000:53999]',
                        '[54000:59999]',
                    ],
                    'estimated_mae': [
                        850.416457004193,
                        849.5344545206443,
                        842.6525418315433,
                        848.7975855207446,
                        843.5694783751043,
                        715.2613958939438,
                        713.1295215575552,
                        711.792248432539,
                        714.6995666403693,
                        704.6573802593545,
                    ],
                    'estimated_mape': [
                        0.23102055597132812,
                        0.23250478545148318,
                        0.23338290531470882,
                        0.2314354653461496,
                        0.23354717477207032,
                        0.25629086361042075,
                        0.2578768766962804,
                        0.257178352802213,
                        0.25710011966564134,
                        0.2597570042079713,
                    ],
                    'estimated_mse': [
                        1138943.3424794453,
                        1130244.7909760615,
                        1111250.668381289,
                        1125011.8168356917,
                        1113510.8315744968,
                        862137.1123436225,
                        862884.8618045567,
                        861027.7984957256,
                        865692.9254567219,
                        845292.4329329217,
                    ],
                    'estimated_rmse': [
                        1067.21288526678,
                        1063.1297150282562,
                        1054.1587491366226,
                        1060.66574227496,
                        1055.23022681048,
                        928.5133883491517,
                        928.9159605715453,
                        927.9158358901553,
                        930.4262063467054,
                        919.3978643291063,
                    ],
                    'estimated_msle': [
                        0.0706487011727066,
                        0.0712183417486575,
                        0.07176176661177244,
                        0.07059396528674712,
                        0.07201370299214875,
                        0.094495149546463,
                        0.09527628885217262,
                        0.09534546951023287,
                        0.09517997014177552,
                        0.09653181184031125,
                    ],
                    'estimated_rmsle': [
                        0.2657982339533252,
                        0.26686764837397864,
                        0.26788386777066747,
                        0.2656952488975802,
                        0.26835369010346916,
                        0.30740063361428355,
                        0.3086685744486676,
                        0.30878061712198335,
                        0.3085125121316403,
                        0.3106956900896941,
                    ],
                }
            ),
        ),
        (
            {'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': [
                        '[0:5999]',
                        '[6000:11999]',
                        '[12000:17999]',
                        '[18000:23999]',
                        '[24000:29999]',
                        '[30000:35999]',
                        '[36000:41999]',
                        '[42000:47999]',
                        '[48000:53999]',
                        '[54000:59999]',
                    ],
                    'estimated_mae': [
                        850.416457004193,
                        849.5344545206443,
                        842.6525418315433,
                        848.7975855207446,
                        843.5694783751043,
                        715.2613958939438,
                        713.1295215575552,
                        711.792248432539,
                        714.6995666403693,
                        704.6573802593545,
                    ],
                    'estimated_mape': [
                        0.23102055597132812,
                        0.23250478545148318,
                        0.23338290531470882,
                        0.2314354653461496,
                        0.23354717477207032,
                        0.25629086361042075,
                        0.2578768766962804,
                        0.257178352802213,
                        0.25710011966564134,
                        0.2597570042079713,
                    ],
                    'estimated_mse': [
                        1138943.3424794453,
                        1130244.7909760615,
                        1111250.668381289,
                        1125011.8168356917,
                        1113510.8315744968,
                        862137.1123436225,
                        862884.8618045567,
                        861027.7984957256,
                        865692.9254567219,
                        845292.4329329217,
                    ],
                    'estimated_rmse': [
                        1067.21288526678,
                        1063.1297150282562,
                        1054.1587491366226,
                        1060.66574227496,
                        1055.23022681048,
                        928.5133883491517,
                        928.9159605715453,
                        927.9158358901553,
                        930.4262063467054,
                        919.3978643291063,
                    ],
                    'estimated_msle': [
                        0.0706487011727066,
                        0.0712183417486575,
                        0.07176176661177244,
                        0.07059396528674712,
                        0.07201370299214875,
                        0.094495149546463,
                        0.09527628885217262,
                        0.09534546951023287,
                        0.09517997014177552,
                        0.09653181184031125,
                    ],
                    'estimated_rmsle': [
                        0.26579823395332525,
                        0.2668676483739787,
                        0.2678838677706675,
                        0.2656952488975803,
                        0.26835369010346916,
                        0.3074006336142836,
                        0.3086685744486676,
                        0.30878061712198335,
                        0.3085125121316404,
                        0.3106956900896941,
                    ],
                }
            ),
        ),
    ],
    ids=[
        'size_based_without_timestamp',
        'size_based_with_timestamp',
        'count_based_without_timestamp',
        'count_based_with_timestamp',
        'period_based_with_timestamp',
        'default_without_timestamp',
        'default_with_timestamp',
    ],
)
def test_dle_for_regression_with_timestamps(calculator_opts, expected):
    ref_df, ana_df, _ = load_synthetic_car_price_dataset()
    dle = DLE(
        feature_column_names=[col for col in ref_df.columns if col not in ['timestamp', 'y_true', 'y_pred']],
        y_pred='y_pred',
        y_true='y_true',
        metrics=['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle'],
        **calculator_opts,
    ).fit(ref_df)
    result = dle.estimate(ana_df)
    sut = result.filter(period='analysis').to_df()[
        [('chunk', 'key')] + [(metric.column_name, 'value') for metric in result.metrics]
    ]
    sut.columns = [
        'key',
        'estimated_mae',
        'estimated_mape',
        'estimated_mse',
        'estimated_rmse',
        'estimated_msle',
        'estimated_rmsle',
    ]

    pd.testing.assert_frame_equal(expected, sut)


@pytest.mark.parametrize('metric_cls', [MAE, MAPE, MSE, MSLE, RMSE, RMSLE])
def test_method_logs_warning_when_lower_threshold_is_overridden_by_metric_limits(caplog, metric_cls):
    ref_df, ana_df, _ = load_synthetic_car_price_dataset()

    metric = metric_cls(
        feature_column_names=[
            col for col in ref_df.columns if col not in ['timestamp', 'y_true', 'y_pred', 'fuel', 'transmission']
        ],
        y_pred='y_pred',
        y_true='y_true',
        chunker=DefaultChunker(),
        threshold=ConstantThreshold(lower=-1),
        tune_hyperparameters=False,
        hyperparameters={},
        hyperparameter_tuning_config={
            "time_budget": 15,
            "metric": "mse",
            "estimator_list": ['lgbm'],
            "eval_method": "cv",
            "hpo_method": "cfo",
            "n_splits": 5,
            "task": 'regression',
            "seed": 1,
            "verbose": 0,
        },
    )
    metric.fit(ref_df)

    assert (
        f'{metric.display_name} lower threshold value -1 overridden by '
        f'lower threshold value limit {metric.lower_threshold_value_limit}' in caplog.messages
    )
