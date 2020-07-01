import pytest

from df_and_order.df_transform import DfTransformConfig, TRANSFORM_DF_FORMAT_KEY, TRANSFORM_IN_MEMORY_KEY, \
    TRANSFORM_PERMANENT_KEY, TRANSFORM_SOURCE_ID_KEY, TRANSFORM_SOURCE_IN_MEMORY_KEY
from df_and_order.df_transform_state import DfTransformState, TRANSFORM_STATE_SOURCE_KEY
from df_and_order.df_transform_step import DfTransformStepConfig, TRANSFORM_STEP_MODULE_PATH_KEY, \
    TRANSFORM_STEP_PARAMS_KEY


def test_properties(mocker):
    transform = mocker.Mock()
    source_transform = mocker.Mock()

    state = DfTransformState(transform=transform,
                             source_transform=source_transform)

    assert state.transform == transform
    assert state.source_transform == source_transform


@pytest.mark.parametrize("use_source", [True, False], ids=['source', 'no_source'])
def test_to_dict(use_source):
    source_transform_id = 'source_trans_id'
    transform_id = 'trans_id'
    df_format = 'df_format'
    in_memory_steps = [
        DfTransformStepConfig(module_path='inmem1', params={'a': 1}),
        DfTransformStepConfig(module_path='inmem2', params={'b': 2}),
    ]
    permanent_steps = [
        DfTransformStepConfig(module_path='perm3', params={}),
        DfTransformStepConfig(module_path='perm4', params={'c': 3}),
    ]
    source_transform = None
    if use_source:
        source_transform = DfTransformConfig(transform_id=source_transform_id,
                                             df_format=df_format,
                                             in_memory_steps=in_memory_steps)

    transform = DfTransformConfig(transform_id=transform_id,
                                  source_id=source_transform_id if use_source else None,
                                  df_format=df_format,
                                  permanent_steps=permanent_steps)

    state = DfTransformState(transform=transform,
                             source_transform=source_transform)

    res_transform_id, res_state_dict = state.to_dict()

    test_transform_dict = transform.to_dict()[1]
    test_state_dict = test_transform_dict
    if use_source:
        test_source_dict = source_transform.to_dict()[1]
        test_transform_dict[TRANSFORM_STATE_SOURCE_KEY] = test_source_dict

    assert res_transform_id == transform_id
    assert res_state_dict == test_state_dict


@pytest.mark.parametrize("use_source", [True, False], ids=['source', 'no_source'])
def test_from_dict(use_source):
    source_id = 'test_source_id' if use_source else None
    df_format = 'format'
    in_memory_steps = [
        {
            TRANSFORM_STEP_MODULE_PATH_KEY: 'inmem1',
            TRANSFORM_STEP_PARAMS_KEY: {'a': 1}
        },
        {
            TRANSFORM_STEP_MODULE_PATH_KEY: 'inmem2',
            TRANSFORM_STEP_PARAMS_KEY: {'b': 2}
        }
    ]
    permanent_steps = [
        {
            TRANSFORM_STEP_MODULE_PATH_KEY: 'perm3',
        },
        {
            TRANSFORM_STEP_MODULE_PATH_KEY: 'perm4',
            TRANSFORM_STEP_PARAMS_KEY: {'c': 3}
        }
    ]
    transform_dict = {
        TRANSFORM_DF_FORMAT_KEY: df_format,
        TRANSFORM_IN_MEMORY_KEY: in_memory_steps,
    }
    state_dict = transform_dict
    source_transform_dict = None
    if use_source:
        source_transform_dict = {
            TRANSFORM_DF_FORMAT_KEY: df_format,
            TRANSFORM_PERMANENT_KEY: permanent_steps
        }
        state_dict[TRANSFORM_STATE_SOURCE_KEY] = source_transform_dict
        state_dict[TRANSFORM_SOURCE_ID_KEY] = source_id

    transform_id = 'trans_id'
    state = DfTransformState.from_dict(transform_id=transform_id,
                                       state_dict=state_dict)

    test_transform = DfTransformConfig.from_dict(transform_id=transform_id, transform_dict=transform_dict)
    assert state.transform == test_transform

    if use_source:
        test_source_transform = DfTransformConfig.from_dict(transform_id=source_id,
                                                            transform_dict=source_transform_dict)
        assert state.source_transform == test_source_transform
