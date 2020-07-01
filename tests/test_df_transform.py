import pytest

from df_and_order.df_transform import DfTransformConfig, TRANSFORM_IN_MEMORY_KEY, TRANSFORM_PERMANENT_KEY, \
    TRANSFORM_SOURCE_IN_MEMORY_KEY, TRANSFORM_SOURCE_ID_KEY, TRANSFORM_DF_FORMAT_KEY
from df_and_order.df_transform_step import DfTransformStepConfig, TRANSFORM_STEP_MODULE_PATH_KEY, \
    TRANSFORM_STEP_PARAMS_KEY


def test_properties(mocker):
    transform_id = 'trans_id'
    df_format = 'format'
    source_id = 'source_id'
    source_in_memory_steps = mocker.Mock()
    in_memory_steps = mocker.Mock()
    permanent_steps = mocker.Mock()

    transform = DfTransformConfig(transform_id=transform_id,
                                  source_id=source_id,
                                  df_format=df_format,
                                  source_in_memory_steps=source_in_memory_steps,
                                  in_memory_steps=in_memory_steps,
                                  permanent_steps=permanent_steps)

    assert transform.transform_id == transform_id
    assert transform.df_format == df_format
    assert transform.source_id == source_id
    assert transform.source_in_memory_steps == source_in_memory_steps
    assert transform.in_memory_steps == in_memory_steps
    assert transform.permanent_steps == permanent_steps


def test_eq(mocker):
    transform_id = 'trans_id'
    df_format = 'format'
    source_id = 'source_id'
    source_in_memory_steps = mocker.Mock()
    in_memory_steps = mocker.Mock()
    permanent_steps = mocker.Mock()

    transform = DfTransformConfig(transform_id=transform_id,
                                  source_id=source_id,
                                  df_format=df_format,
                                  source_in_memory_steps=source_in_memory_steps,
                                  in_memory_steps=in_memory_steps,
                                  permanent_steps=permanent_steps)

    another_transform = DfTransformConfig(transform_id=transform_id,
                                          source_id=source_id,
                                          df_format=df_format,
                                          source_in_memory_steps=source_in_memory_steps,
                                          in_memory_steps=in_memory_steps,
                                          permanent_steps=permanent_steps)

    assert transform == another_transform

    another_transform._transform_id = 'bad'

    assert transform != another_transform

@pytest.mark.parametrize("use_source_id", [True, False], ids=['source_id', 'no_source_id'])
def test_to_dict(use_source_id):
    transform_id = 'trans_id'
    df_format = 'df_format'
    source_id = 'test_source_id' if use_source_id else None
    source_in_memory_steps = [
        DfTransformStepConfig(module_path='source_inmem1', params={'_a': 1}),
        DfTransformStepConfig(module_path='source_inmem2', params={'_b': 2}),
    ] if use_source_id else None
    in_memory_steps = [
        DfTransformStepConfig(module_path='inmem1', params={'a': 1}),
        DfTransformStepConfig(module_path='inmem2', params={'b': 2}),
    ]
    permanent_steps = [
        DfTransformStepConfig(module_path='perm3', params={}),
        DfTransformStepConfig(module_path='perm4', params={'c': 3}),
    ]

    transform = DfTransformConfig(transform_id=transform_id,
                                  df_format=df_format,
                                  source_id=source_id,
                                  source_in_memory_steps=source_in_memory_steps,
                                  in_memory_steps=in_memory_steps,
                                  permanent_steps=permanent_steps)

    res_transform_id, res_transform_dict = transform.to_dict()
    res_source_id = res_transform_dict.get(TRANSFORM_SOURCE_ID_KEY)
    test_transform_dict = {
        TRANSFORM_DF_FORMAT_KEY: df_format,
        TRANSFORM_IN_MEMORY_KEY: [
            {
                TRANSFORM_STEP_MODULE_PATH_KEY: 'inmem1',
                TRANSFORM_STEP_PARAMS_KEY: {'a': 1}
            },
            {
                TRANSFORM_STEP_MODULE_PATH_KEY: 'inmem2',
                TRANSFORM_STEP_PARAMS_KEY: {'b': 2}
            }
        ],
        TRANSFORM_PERMANENT_KEY: [
            {
                TRANSFORM_STEP_MODULE_PATH_KEY: 'perm3',
            },
            {
                TRANSFORM_STEP_MODULE_PATH_KEY: 'perm4',
                TRANSFORM_STEP_PARAMS_KEY: {'c': 3}
            }
        ]
    }

    if use_source_id:
        test_transform_dict[TRANSFORM_SOURCE_ID_KEY] = source_id
        test_transform_dict[TRANSFORM_SOURCE_IN_MEMORY_KEY] = [
            {
                TRANSFORM_STEP_MODULE_PATH_KEY: 'source_inmem1',
                TRANSFORM_STEP_PARAMS_KEY: {'_a': 1}
            },
            {
                TRANSFORM_STEP_MODULE_PATH_KEY: 'source_inmem2',
                TRANSFORM_STEP_PARAMS_KEY: {'_b': 2}
            }
        ]

    assert res_transform_id == transform_id
    assert res_source_id == source_id
    assert res_transform_dict == test_transform_dict


@pytest.mark.parametrize("use_source_id", [True, False], ids=['source_id', 'no_source_id'])
def test_from_dict(use_source_id):
    source_id = 'test_source_id' if use_source_id else None
    df_format = 'format'
    source_in_memory_steps = [
        {
            TRANSFORM_STEP_MODULE_PATH_KEY: 'source_inmem1',
            TRANSFORM_STEP_PARAMS_KEY: {'_a': 1}
        },
        {
            TRANSFORM_STEP_MODULE_PATH_KEY: 'source_inmem2',
            TRANSFORM_STEP_PARAMS_KEY: {'_b': 2}
        }
    ] if use_source_id else None
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
        TRANSFORM_PERMANENT_KEY: permanent_steps
    }

    if use_source_id:
        transform_dict[TRANSFORM_SOURCE_ID_KEY] = source_id
        transform_dict[TRANSFORM_SOURCE_IN_MEMORY_KEY] = source_in_memory_steps

    transform_id = 'trans_id'
    transform = DfTransformConfig.from_dict(transform_id=transform_id,
                                            transform_dict=transform_dict)

    assert transform.df_format == df_format
    assert transform.transform_id == transform_id
    if use_source_id:
        for original_dict, step in zip(source_in_memory_steps, transform.source_in_memory_steps):
            assert original_dict[TRANSFORM_STEP_MODULE_PATH_KEY] == step.module_path
            assert original_dict[TRANSFORM_STEP_PARAMS_KEY] == step.params

    for original_dict, step in zip(in_memory_steps, transform.in_memory_steps):
        assert original_dict[TRANSFORM_STEP_MODULE_PATH_KEY] == step.module_path
        assert original_dict[TRANSFORM_STEP_PARAMS_KEY] == step.params

    for original_dict, step in zip(permanent_steps, transform.permanent_steps):
        assert original_dict[TRANSFORM_STEP_MODULE_PATH_KEY] == step.module_path
        if original_dict.get(TRANSFORM_STEP_PARAMS_KEY):
            assert original_dict[TRANSFORM_STEP_PARAMS_KEY] == step.params
