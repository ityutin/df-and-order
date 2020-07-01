import pytest

from df_and_order.df_transform_step import TRANSFORM_STEP_MODULE_PATH_KEY, TRANSFORM_STEP_PARAMS_KEY, \
    DfTransformStepConfig
from tests.dates_transform import TestDatesTransformStep


def test_from_step_type():
    step_type = TestDatesTransformStep
    params = {'cols': ['a']}

    step_config = DfTransformStepConfig.from_step_type(step_type=step_type, params=params)

    assert step_config.module_path == 'tests.dates_transform.TestDatesTransformStep'
    assert step_config.params == params


def test_eq():
    module_path = 'some/path'
    params = {'param1': 1, 'param2': 'value'}
    step_dict = {
        TRANSFORM_STEP_MODULE_PATH_KEY: module_path,
        TRANSFORM_STEP_PARAMS_KEY: params
    }

    step_config = DfTransformStepConfig.from_dict(step_dict=step_dict)
    another_step_config = DfTransformStepConfig.from_dict(step_dict=step_dict.copy())

    assert step_config == another_step_config

    another_step_config.module_path = 'bad'

    assert step_config != another_step_config


def test_from_dict():
    module_path = 'some/path'
    params = {'param1': 1, 'param2': 'value'}
    step_dict = {
        TRANSFORM_STEP_MODULE_PATH_KEY: module_path,
        TRANSFORM_STEP_PARAMS_KEY: params
    }

    step_config = DfTransformStepConfig.from_dict(step_dict=step_dict)

    assert step_config.module_path == module_path
    assert step_config.params == params


def test_to_dict():
    module_path = 'some/path'
    params = {'param1': 1, 'param2': 'value'}
    step_config = DfTransformStepConfig(module_path=module_path, params=params)

    step_dict = step_config.to_dict()

    assert step_dict[TRANSFORM_STEP_MODULE_PATH_KEY] == module_path
    assert step_dict[TRANSFORM_STEP_PARAMS_KEY] == params

    step_config = DfTransformStepConfig(module_path=module_path, params={})
    step_dict = step_config.to_dict()

    assert step_dict.get(TRANSFORM_STEP_PARAMS_KEY) is None
