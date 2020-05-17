import pytest

from df_and_order.df_transform_step import TRANSFORM_STEP_MODULE_PATH_KEY, TRANSFORM_STEP_PARAMS_KEY, \
    DfTransformStepConfig


def test_from_dict():
    module_path = 'some/path'
    params = {'param1': 1, 'param2': 'value'}
    step_dict = {
        TRANSFORM_STEP_MODULE_PATH_KEY: module_path,
        TRANSFORM_STEP_PARAMS_KEY: params
    }

    step = DfTransformStepConfig.from_dict(step_dict=step_dict)

    assert step.module_path == module_path
    assert step.params == params


def test_to_dict():
    module_path = 'some/path'
    params = {'param1': 1, 'param2': 'value'}
    step = DfTransformStepConfig(module_path=module_path, params=params)

    step_dict = step.to_dict()

    assert step_dict[TRANSFORM_STEP_MODULE_PATH_KEY] == module_path
    assert step_dict[TRANSFORM_STEP_PARAMS_KEY] == params

    step = DfTransformStepConfig(module_path=module_path, params={})
    step_dict = step.to_dict()

    assert step_dict.get(TRANSFORM_STEP_PARAMS_KEY) is None
