import numpy as np
import pytest

from nannyml.exceptions import InvalidArgumentsException, ThresholdException
from nannyml.thresholds import ConstantThreshold, StandardDeviationThreshold


@pytest.mark.parametrize('lower, upper', [(0.0, 1.0), (0, 1), (-1, 1), (None, 1.0), (0.1, None), (None, None)])
def test_constant_threshold_init_sets_instance_attributes(lower, upper):
    sut = ConstantThreshold(lower, upper)

    assert sut.lower == lower
    assert sut.upper == upper


def test_constant_threshold_init_sets_default_instance_attributes():
    sut = ConstantThreshold()

    assert sut.lower is None
    assert sut.upper is None


@pytest.mark.parametrize('lower, upper', [(1.0, 0.0), (0.0, -1.0), (2.1, 2.1)])
def test_constant_threshold_init_raises_threshold_exception_when_breaking_lower_upper_strict_order(lower, upper):
    with pytest.raises(ThresholdException, match=f"lower threshold {lower} must be less than upper threshold {upper}"):
        _ = ConstantThreshold(lower, upper)


@pytest.mark.parametrize(
    'lower, upper, param, param_type',
    [
        ('0.0', 1.0, 'lower', 'str'),
        (0.0, '1.0', 'upper', 'str'),
        (True, 1.0, 'lower', 'bool'),
        (0.0, True, 'upper', 'bool'),
        (0.0, {}, 'upper', 'dict'),
    ],
)
def test_constant_threshold_init_raises_invalid_arguments_exception_when_given_wrongly_typed_arguments(
    upper, lower, param, param_type
):
    with pytest.raises(
        InvalidArgumentsException,
        match=f"expected type of '{param}' to be 'float', 'int' or None but got '{param_type}'",
    ):
        _ = ConstantThreshold(lower, upper)


@pytest.mark.parametrize('lower, upper', [(0.0, 1.0), (0, 1), (-1, 1), (None, 1.0), (0.1, None), (None, None)])
def test_constant_threshold_returns_correct_threshold_values(lower, upper):
    t = ConstantThreshold(lower, upper)
    lt, ut = t.thresholds(np.ndarray(range(10)))

    assert lt == lower
    assert ut == upper


@pytest.mark.parametrize(
    'lower_multiplier, upper_multiplier, offset_from',
    [(1, 1, np.median), (1, None, np.median), (None, 1, np.median), (None, None, np.median)],
)
def test_standard_deviation_threshold_init_sets_instance_attributes(lower_multiplier, upper_multiplier, offset_from):
    sut = StandardDeviationThreshold(lower_multiplier, upper_multiplier, offset_from)

    assert sut.std_lower_multiplier == lower_multiplier
    assert sut.std_upper_multiplier == upper_multiplier
    assert sut.offset_from == offset_from


def test_standard_deviation_threshold_init_sets_default_instance_attributes():
    sut = StandardDeviationThreshold()

    assert sut.std_lower_multiplier == 3
    assert sut.std_upper_multiplier == 3
    assert sut.offset_from == np.mean


@pytest.mark.parametrize(
    'lower_multiplier, upper_multiplier, param, param_type',
    [
        ('0.0', 1.0, 'std_lower_multiplier', 'str'),
        (0.0, '1.0', 'std_upper_multiplier', 'str'),
        (True, 1.0, 'std_lower_multiplier', 'bool'),
        (0.0, True, 'std_upper_multiplier', 'bool'),
        (0.0, {}, 'std_upper_multiplier', 'dict'),
    ],
)
def test_standard_deviation_threshold_init_raises_invalid_arguments_exception_when_given_wrongly_typed_arguments(
    lower_multiplier, upper_multiplier, param, param_type
):
    with pytest.raises(
        InvalidArgumentsException,
        match=f"expected type of '{param}' to be 'float', 'int' or None but got '{param_type}'",
    ):
        _ = StandardDeviationThreshold(std_lower_multiplier=lower_multiplier, std_upper_multiplier=upper_multiplier)


@pytest.mark.parametrize('offset_from, expected', [(np.min, -1), (np.max, 1), (np.median, 0), (np.mean, 0)])
def test_standard_deviation_threshold_applies_offset_from(offset_from, expected):
    t = StandardDeviationThreshold(std_lower_multiplier=0, std_upper_multiplier=0, offset_from=offset_from)

    lt, ut = t.thresholds(np.asarray([-1, -0.5, 0, 0.5, 1]))

    assert lt == expected
    assert ut == expected


@pytest.mark.parametrize(
    'std_lower_multiplier, expected_threshold', [(1, -1.8660254037844386), (0, -1), (2, -2.732050807568877)]
)
def test_standard_deviation_threshold_correctly_applies_std_lower_multiplier(std_lower_multiplier, expected_threshold):
    t = StandardDeviationThreshold(std_lower_multiplier=std_lower_multiplier, offset_from=np.min)
    lt, _ = t.thresholds(np.asarray([-1, 1, 1, 1]))
    assert lt == expected_threshold


@pytest.mark.parametrize(
    'std_lower_multiplier, std_upper_multiplier, exp_lower_threshold, exp_upper_threshold',
    [(None, 0, None, -1.0), (0, None, -1.0, None), (None, None, None, None)],
)
def test_standard_deviation_threshold_treats_none_multiplier_as_no_threshold(
    std_lower_multiplier, std_upper_multiplier, exp_lower_threshold, exp_upper_threshold
):
    t = StandardDeviationThreshold(std_lower_multiplier, std_upper_multiplier, offset_from=np.min)
    lt, ut = t.thresholds(np.asarray([-1, 1, 1, 1]))

    assert lt == exp_lower_threshold
    assert ut == exp_upper_threshold


@pytest.mark.parametrize(
    'low_mult, up_mult, offset_from, exp_low_threshold, exp_up_threshold',
    [
        (1.4, 2, np.median, 2.382381972241136, 31.81088289679838),
        (0.3, 3.1, np.min, -2.5966324345197567, 26.83186849003749),
    ],
)
def test_standard_deviation_threshold_correctly_returns_thresholds(
    low_mult, up_mult, offset_from, exp_low_threshold, exp_up_threshold
):
    t = StandardDeviationThreshold(low_mult, up_mult, offset_from)
    lt, ut = t.thresholds(np.asarray(range(30)))

    assert lt == exp_low_threshold
    assert ut == exp_up_threshold


def test_standard_deviation_threshold_raises_threshold_exception_when_negative_lower_multiplier_given():
    with pytest.raises(ThresholdException, match="'std_lower_multiplier' should be greater than 0 but got value -1"):
        _ = StandardDeviationThreshold(-1, 0)


def test_standard_deviation_threshold_raises_threshold_exception_when_negative_upper_multiplier_given():
    with pytest.raises(ThresholdException, match="'std_upper_multiplier' should be greater than 0 but got value -1"):
        _ = StandardDeviationThreshold(0, -1)
