import pytest
import pandas as pd
import numpy as np
from df_and_order.df_config import DfConfig, DF_ID_KEY, DF_INITIAL_FORMAT_KEY
from df_and_order.df_reader import DfReader
from df_and_order.df_transform import DfTransformConfig, TRANSFORM_IN_MEMORY_KEY
from df_and_order.df_transform_state import TRANSFORM_STATE_SOURCE_KEY
from df_and_order.df_transform_step import DfTransformStepConfig, DfTransformStep
from df_and_order.helpers import FileInspector
from tests.test_df_cache import TestDfCache
from tests.test_df_config import _get_config


@pytest.fixture()
def df_id():
    return 'test_id'


@pytest.fixture()
def test_path():
    return 'path/to/config/'


@pytest.fixture
def transformed_format():
    return 'trans_format'


@pytest.fixture()
def initial_df():
    result = pd.DataFrame({
        'a': ['2020-02-01', '2020-02-02', '2020-02-03'],
        'b': [1, 2, 3],
        'c': ['2020-01-01', '2020-01-02', '2020-01-03']
    })
    return result


def build_config(mocker, df_id, test_path, initial_format):
    config_dict = {
        DF_ID_KEY: df_id,
        DF_INITIAL_FORMAT_KEY: initial_format,
    }

    result = _get_config(mocker=mocker, df_id=df_id, dir_path=test_path, config_dict=config_dict)
    return result


@pytest.mark.parametrize("transform_id", ['trans_1', None], ids=['trans', 'no_trans'])
@pytest.mark.parametrize("valid_case", [True, False], ids=['valid', 'not_valid'])
def test_df_exists(mocker, df_id, test_path, transform_id, valid_case, transformed_format):
    mocker.patch.object(DfConfig, DfConfig.config_exists.__name__, lambda dir_path: True)
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format='init_format')
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)

    if transform_id:
        filename = f'{transform_id}_{df_id}.{transformed_format}'
    else:
        filename = f'{df_id}.{config.initial_df_format}'

    if not valid_case:
        filename += 'garbage'

    test_file_path = f'{test_path}{df_id}/{filename}'
    mocker.patch.object(DfReader, DfReader._is_file_exists.__name__, lambda path: path == test_file_path)

    reader = DfReader(dir_path=test_path, format_to_cache_map={})
    transform = DfTransformConfig(transform_id=transform_id,
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                      DfTransformStepConfig(
                                          module_path='tests.drop_cols_transform.TestDropColsTransformStep',
                                          params={'cols_to_drop': ['b']}),
                                  ])
    if transform_id:
        # do nothing when it comes to save the config
        mocker.patch.object(DfConfig, DfConfig._save.__name__)
        reader.register_transform(df_id=df_id,
                                  df_config=config,
                                  transform=transform)
    if valid_case:
        assert reader.df_exists(df_id=df_id, transform_id=transform_id)
    else:
        assert not reader.df_exists(df_id=df_id, transform_id=transform_id)


def test_create_df_config(mocker, df_id, test_path):
    reader = DfReader(dir_path=test_path, format_to_cache_map={})

    initial_format = mocker.Mock()
    metadata = mocker.Mock()
    transform = mocker.Mock()

    config_create_mock = mocker.patch.object(DfConfig, DfConfig.create_config.__name__)

    reader.create_df_config(df_id=df_id,
                            initial_df_format=initial_format,
                            metadata=metadata,
                            transform=transform)

    dir_path = _df_dir_path(dir_path=test_path, df_id=df_id)
    config_create_mock.assert_called_with(dir_path=dir_path,
                                          df_id=df_id,
                                          initial_df_format=initial_format,
                                          metadata=metadata,
                                          transform=transform)


def test_register_transform(mocker, df_id, test_path, transformed_format):
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format='init_format')
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    register_transform_mock = mocker.patch.object(DfConfig, DfConfig.register_transform.__name__)

    transform = mocker.Mock()
    transform.transform_id = 'trans_id'
    transform.df_format = transformed_format
    filename = f'{transform.transform_id}_{df_id}.{transformed_format}'

    reader = DfReader(dir_path=test_path, format_to_cache_map={})
    reader.register_transform(df_id=df_id,
                              df_config=config,
                              transform=transform)

    register_transform_mock.assert_called_with(transform=transform,
                                               filename=filename)


def test_read_both_transform_id_and_transform(mocker, df_id, test_path):
    reader = DfReader(dir_path=test_path, format_to_cache_map={})
    with pytest.raises(Exception):
        reader.read(df_id=df_id,
                    transform_id='',
                    transform=mocker.Mock())


def test_read_initial(mocker, df_id, test_path, initial_df):
    init_format = 'init_format'
    init_cache = TestDfCache()
    load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    load_mock.return_value = initial_df
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache
    })
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    mocker.patch.object(reader, reader._save_transforms_state_file.__name__)
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    df_dir_path = _df_dir_path(dir_path=test_path, df_id=df_id)
    df_path = f'{df_dir_path}/{df_id}.{config.initial_df_format}'

    test_initial_df = reader.read(df_id=df_id)

    load_mock.assert_called_with(path=df_path)

    assert initial_df.equals(test_initial_df)


@pytest.mark.parametrize("permanent", [True, False], ids=['permanent', 'not_permanent'])
@pytest.mark.parametrize("from_scratch", [True, False], ids=['from_scratch', 'not_from_scratch'])
def test_read_transformed(mocker, df_id, test_path, permanent, from_scratch, initial_df):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    trans_save_mock = mocker.patch.object(transform_cache, transform_cache.save.__name__)
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    transformed_format = 'trans_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    # setup transform

    permanent_steps = []
    if permanent:
        permanent_steps = [
            DfTransformStepConfig(module_path='tests.drop_cols_transform.TestDropColsTransformStep',
                                  params={'cols_to_drop': ['b']}),
        ]

    transform = DfTransformConfig(transform_id='transform_id',
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                    DfTransformStepConfig(module_path='tests.zero_transform.TestZeroTransformStep',
                                                          params={'zero_cols': ['a']}),
                                    DfTransformStepConfig(module_path='tests.dates_transform.TestDatesTransformStep',
                                                          params={'dates_cols': ['c']})
                                  ],
                                  permanent_steps=permanent_steps)

    # setup config

    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    # ignore states for this test
    mocker.patch.object(reader, reader._save_transforms_state_file.__name__)
    mocker.patch.object(reader, reader._try_validate_transform.__name__)
    #
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)
    # override the config getter
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    # just overriding it for the test environment
    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, lambda file_path: 1.0)

    df_dir_path = _df_dir_path(dir_path=test_path, df_id=df_id)

    if not from_scratch:
        # if not from scratch we need to stub cached df
        mocker.patch.object(DfReader, DfReader.df_exists.__name__, lambda _, df_id, transform_id: True)

        transformed_df = initial_df.copy()
        for step_config in transform.permanent_steps:
            step = DfTransformStep.build_transform(config=step_config)
            transformed_df = step.transform(df=transformed_df)

        trans_load_mock = mocker.patch.object(transform_cache, transform_cache.load.__name__)
        trans_load_mock.return_value = transformed_df

    transformed_df = reader.read(df_id=df_id, transform=transform)

    # checks

    if from_scratch:
        init_df_path = f'{df_dir_path}/{df_id}.{config.initial_df_format}'
        init_load_mock.assert_called_with(path=init_df_path)

    if permanent:
        test_columns = {'a', 'c'}
    else:
        test_columns = {'a', 'b', 'c'}

    assert set(transformed_df.columns) == test_columns
    assert transformed_df['a'].unique()[0] == 0
    assert transformed_df.select_dtypes(include=[np.datetime64]).columns[0] == 'c'

    if from_scratch and permanent:
        df_path = f'{df_dir_path}/{transform.transform_id}_{df_id}.{transformed_format}'
        trans_save_mock.assert_called_with(df=transformed_df,
                                           path=df_path)


@pytest.mark.parametrize("is_df_outdated", [True, False], ids=['outdated', 'not_outdated'])
@pytest.mark.parametrize("forced", [True, False], ids=['forced', 'not_forced'])
def test_read_transformed_check_ts(mocker, df_id, test_path, initial_df, is_df_outdated, forced):
    # if not from scratch we need to stub cached df
    zero_module_path = 'tests.zero_transform.TestZeroTransformStep'
    zero_last_modified_ts = 1.0
    drop_module_path = 'tests.drop_cols_transform.TestDropColsTransformStep'
    drop_last_modified_ts = 2.0
    if is_df_outdated:
        df_last_modified_ts = min(zero_last_modified_ts, drop_last_modified_ts) - 1.0
    else:
        df_last_modified_ts = max(zero_last_modified_ts, drop_last_modified_ts) + 1.0

    mocker.patch.object(DfReader,
                        DfReader.df_exists.__name__,
                        lambda _, df_id, transform_id: True)
    mocker.patch.object(DfReader,
                        DfReader._df_last_modified_ts.__name__,
                        lambda _, df_id, transform_id: df_last_modified_ts)

    transformed_format = 'transformed_format'
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format='init_format')
    # do nothing when it comes to save config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    # override config getter
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)

    permanent_steps = [
        DfTransformStepConfig(module_path=zero_module_path,
                              params={'zero_cols': ['a']}),
        DfTransformStepConfig(module_path=drop_module_path,
                              params={'cols_to_drop': ['b']}),
    ]

    transform = DfTransformConfig(transform_id='transform_id',
                                  df_format=transformed_format,
                                  in_memory_steps=[],
                                  permanent_steps=permanent_steps)

    trans_cache = TestDfCache()
    trans_load_mock = mocker.patch.object(trans_cache, trans_cache.load.__name__)
    trans_df = mocker.Mock()
    trans_load_mock.return_value = trans_df
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        transformed_format: trans_cache
    })
    # ignore states for this test
    mocker.patch.object(reader, reader._save_transforms_state_file.__name__)
    mocker.patch.object(reader, reader._try_validate_transform.__name__)
    #
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)

    def last_modified_date(file_path: str):
        if 'zero_transform' in file_path:
            return zero_last_modified_ts
        elif 'drop_cols_transform' in file_path:
            return drop_last_modified_ts
        else:
            raise ValueError('???')

    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, last_modified_date)

    if is_df_outdated and not forced:
        with pytest.raises(Exception):
            reader.read(df_id=df_id, transform=transform, forced=forced)
    else:
        res_trans_df = reader.read(df_id=df_id, transform=transform, forced=forced)
        assert res_trans_df == trans_df


def test_in_memory_shape_when_permanent_is_present(mocker, df_id, test_path, initial_df, transformed_format):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    # setup transforms
    drop_cols_module_path = 'tests.drop_cols_transform.TestDropColsTransformStep'
    dates_module_path = 'tests.dates_transform.TestDatesTransformStep'
    transform = DfTransformConfig(transform_id='transform_id',
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                      DfTransformStepConfig(module_path=drop_cols_module_path,
                                                            params={'cols_to_drop': ['a']}),
                                  ],
                                  permanent_steps=[
                                      DfTransformStepConfig(module_path=dates_module_path,
                                                            params={'dates_cols': ['c']})
                                  ])

    # setup config
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    # override the config getter
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)

    with pytest.raises(Exception):
        reader.read(df_id=df_id, transform=transform)


def test_source_id(mocker, df_id, test_path, initial_df, transformed_format):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    # setup transforms
    drop_cols_module_path = 'tests.drop_cols_transform.TestDropColsTransformStep'
    dates_module_path = 'tests.dates_transform.TestDatesTransformStep'
    zero_module_path = 'tests.zero_transform.TestZeroTransformStep'
    source_transform_id = 'source_transform_id'
    source_transform = DfTransformConfig(transform_id=source_transform_id,
                                         df_format=transformed_format,
                                         in_memory_steps=[
                                             DfTransformStepConfig(module_path=dates_module_path,
                                                                   params={'dates_cols': ['c']})
                                         ])
    transform = DfTransformConfig(transform_id='test_transform_id',
                                  df_format=transformed_format,
                                  source_id=source_transform_id,
                                  source_in_memory_steps=[
                                      DfTransformStepConfig(module_path=drop_cols_module_path,
                                                            params={'cols_to_drop': ['a']})
                                  ], in_memory_steps=[
                                      DfTransformStepConfig(module_path=zero_module_path,
                                                            params={'zero_cols': []})
                                  ])

    transformed_df = initial_df.copy()
    for step_config in source_transform.in_memory_steps:
        step = DfTransformStep.build_transform(config=step_config)
        transformed_df = step.transform(df=transformed_df)

    trans_load_mock = mocker.patch.object(transform_cache, transform_cache.load.__name__)
    trans_load_mock.return_value = transformed_df

    # setup config
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    # override the config getter
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    # ignore states for this test
    mocker.patch.object(reader, reader._save_transforms_state_file.__name__)
    mocker.patch.object(reader, reader._try_validate_transform.__name__)
    #
    reader.register_transform(df_id=df_id, df_config=config, transform=source_transform)
    transformed_df = reader.read(df_id=df_id, transform=transform)

    assert transformed_df.select_dtypes(include=[np.datetime64]).columns[0] == 'c'
    assert 'a' not in set(transformed_df.columns)


@pytest.mark.parametrize("valid_state", [True, False], ids=['valid_state', 'not_valid_state'])
def test_permanent_transforms_state(mocker, df_id, test_path, valid_state, initial_df, transformed_format):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    # setup transform

    permanent_steps = [
        DfTransformStepConfig(module_path='tests.drop_cols_transform.TestDropColsTransformStep',
                              params={'cols_to_drop': ['b']}),
    ]

    transform = DfTransformConfig(transform_id='transform_id',
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                    DfTransformStepConfig(module_path='tests.zero_transform.TestZeroTransformStep',
                                                          params={'zero_cols': ['a']}),
                                    DfTransformStepConfig(module_path='tests.dates_transform.TestDatesTransformStep',
                                                          params={'dates_cols': ['c']})
                                  ],
                                  permanent_steps=permanent_steps)

    # setup config

    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    mocker.patch.object(reader, DfReader._save_transforms_state_file.__name__)
    transform_dict = transform.to_dict()[1]
    if not valid_state:
        del transform_dict[TRANSFORM_IN_MEMORY_KEY]

    transform_state = {
        transform.transform_id: transform_dict
    }
    mocker.patch.object(DfReader, DfReader._is_file_exists.__name__, lambda path: True)
    mocker.patch.object(reader, DfReader._transforms_state_dicts.__name__, lambda df_id: transform_state)
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, lambda file_path: 1.0)
    mocker.patch.object(DfReader, DfReader.df_exists.__name__, lambda _, df_id, transform_id: True)

    test_transformed_df = initial_df.copy()
    for step_config in transform.permanent_steps:
        step = DfTransformStep.build_transform(config=step_config)
        test_transformed_df = step.transform(df=test_transformed_df)
    trans_load_mock = mocker.patch.object(transform_cache, transform_cache.load.__name__)
    trans_load_mock.return_value = test_transformed_df

    if not valid_state:
        with pytest.raises(Exception):
            reader.read(df_id=df_id, transform=transform)
    else:
        transformed_df = reader.read(df_id=df_id, transform=transform)
        assert transformed_df.equals(test_transformed_df)


@pytest.mark.parametrize("valid_state", [True, False], ids=['valid_state', 'not_valid_state'])
def test_permanent_source_transforms_state(mocker, df_id, test_path, valid_state, initial_df, transformed_format):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    # setup transform
    source_permanent_steps = [
        DfTransformStepConfig(module_path='tests.zero_transform.TestZeroTransformStep',
                              params={'zero_cols': ['a']})
    ]

    permanent_steps = [
        DfTransformStepConfig(module_path='tests.drop_cols_transform.TestDropColsTransformStep',
                              params={'cols_to_drop': ['b']}),
    ]

    source_transform_id = 'source_transform_id'
    source_transform = DfTransformConfig(transform_id=source_transform_id,
                                         df_format=transformed_format,
                                         in_memory_steps=[
                                             DfTransformStepConfig(
                                                 module_path='tests.zero_transform.TestZeroTransformStep',
                                                 params={'cols_to_drop': ['b']}),
                                         ],
                                         permanent_steps=source_permanent_steps)

    transform = DfTransformConfig(transform_id='transform_id',
                                  source_id=source_transform_id,
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                    DfTransformStepConfig(module_path='tests.dates_transform.TestDatesTransformStep',
                                                          params={'dates_cols': ['c']})
                                  ],
                                  permanent_steps=permanent_steps)

    # setup config

    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    mocker.patch.object(reader, DfReader._save_transforms_state_file.__name__)
    transform_dict = transform.to_dict()[1]
    source_transform_dict = source_transform.to_dict()[1]
    if not valid_state:
        del source_transform_dict[TRANSFORM_IN_MEMORY_KEY]

    transform_state = {
        transform.transform_id: transform_dict,
        source_transform.transform_id: source_transform_dict
    }
    transform_dict[TRANSFORM_STATE_SOURCE_KEY] = source_transform_dict
    mocker.patch.object(DfReader, DfReader._is_file_exists.__name__, lambda path: True)
    mocker.patch.object(reader, DfReader._transforms_state_dicts.__name__, lambda df_id: transform_state)
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)
    reader.register_transform(df_id=df_id, df_config=config, transform=source_transform)
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, lambda file_path: 1.0)
    mocker.patch.object(DfReader, DfReader.df_exists.__name__, lambda _, df_id, transform_id: True)

    test_transformed_df = initial_df.copy()
    for step_config in transform.permanent_steps:
        step = DfTransformStep.build_transform(config=step_config)
        test_transformed_df = step.transform(df=test_transformed_df)
    trans_load_mock = mocker.patch.object(transform_cache, transform_cache.load.__name__)
    trans_load_mock.return_value = test_transformed_df

    if not valid_state:
        with pytest.raises(Exception):
            reader.read(df_id=df_id, transform=transform)
    else:
        transformed_df = reader.read(df_id=df_id, transform=transform)
        assert transformed_df.equals(test_transformed_df)


@pytest.mark.parametrize("df_exists", [True, False], ids=['df_exists', 'not_df_exists'])
def test_transforms_state_cleanup(mocker, df_id, test_path, df_exists, initial_df, transformed_format):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df
    transform_load_mock = mocker.patch.object(transform_cache, transform_cache.load.__name__)
    transform_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    transform = DfTransformConfig(transform_id='transform_id',
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                    DfTransformStepConfig(module_path='tests.dates_transform.TestDatesTransformStep',
                                                          params={'dates_cols': ['c']})
                                  ])

    # setup config

    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    mocker.patch.object(reader, DfReader._save_transforms_state_file.__name__)
    transform_dict = transform.to_dict()[1]
    transform_state = {
        transform.transform_id: transform_dict,
    }
    mocker.patch.object(reader, DfReader._transforms_state_dicts.__name__, lambda df_id: transform_state)
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, lambda file_path: 1.0)

    if df_exists:
        mocker.patch.object(DfReader, DfReader.df_exists.__name__, lambda _, df_id, transform_id: True)
        reader.read(df_id=df_id, transform=transform)
    else:
        reader.read(df_id=df_id, transform=transform)
        assert len(transform_state) == 0


def test_transforms_state_save(mocker, df_id, test_path, initial_df, transformed_format):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df
    transform_load_mock = mocker.patch.object(transform_cache, transform_cache.load.__name__)
    transform_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    source_transform_id = 'source_transform_id'
    source_transform = DfTransformConfig(transform_id=source_transform_id,
                                         df_format=transformed_format,
                                         permanent_steps=[
                                             DfTransformStepConfig(module_path='tests.dates_transform.TestDatesTransformStep',
                                                                   params={'dates_cols': ['c']})
                                         ])

    transform = DfTransformConfig(transform_id='transform_id',
                                  source_id=source_transform_id,
                                  df_format=transformed_format,
                                  permanent_steps=[
                                      DfTransformStepConfig(module_path='tests.dates_transform.TestDatesTransformStep',
                                                            params={'dates_cols': ['c']})
                                  ])

    # setup config

    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    save_mock = mocker.patch.object(reader, DfReader._save_transforms_state_file.__name__)
    transform_dict = transform.to_dict()[1]
    transform_dict[TRANSFORM_STATE_SOURCE_KEY] = source_transform.to_dict()[1]
    test_transform_state = {
        transform.transform_id: transform_dict,
    }
    mocker.patch.object(reader, DfReader._transforms_state_dicts.__name__, lambda df_id: {})
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)
    reader.register_transform(df_id=df_id, df_config=config, transform=source_transform)
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, lambda file_path: 1.0)

    reader.read(df_id=df_id, transform=transform)

    save_mock.assert_called_with(df_id=df_id, transforms_state=test_transform_state)


@pytest.mark.parametrize("df_exists", [True, False], ids=['df_exists', 'not_df_exists'])
def test_transforms_state_no_file(mocker, df_id, test_path, df_exists, initial_df, transformed_format):
    # setup df cache
    init_cache = TestDfCache()
    transform_cache = TestDfCache()
    init_load_mock = mocker.patch.object(init_cache, init_cache.load.__name__)
    init_load_mock.return_value = initial_df
    transform_load_mock = mocker.patch.object(transform_cache, transform_cache.load.__name__)
    transform_load_mock.return_value = initial_df

    # setup reader
    init_format = 'init_format'
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        init_format: init_cache,
        transformed_format: transform_cache
    })

    transform = DfTransformConfig(transform_id='transform_id',
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                    DfTransformStepConfig(module_path='tests.dates_transform.TestDatesTransformStep',
                                                          params={'dates_cols': ['c']})
                                  ])

    # setup config

    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format=init_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    mocker.patch.object(reader, DfReader._save_transforms_state_file.__name__)
    transform_dict = transform.to_dict()[1]
    transform_state = {
        transform.transform_id: transform_dict,
    }
    mocker.patch.object(reader, DfReader._transforms_state_dicts.__name__, lambda df_id: transform_state)
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, lambda file_path: 1.0)

    if df_exists:
        mocker.patch.object(DfReader, DfReader.df_exists.__name__, lambda _, df_id, transform_id: True)
        reader.read(df_id=df_id, transform=transform)
    else:
        reader.read(df_id=df_id, transform=transform)
        assert len(transform_state) == 0

@pytest.mark.parametrize("forced", [True, False], ids=['forced', 'not_forced'])
@pytest.mark.parametrize("is_df_outdated", [True, False], ids=['outdated', 'not_outdated'])
def test_transform_state_source_check_ts(mocker, df_id, test_path, initial_df, is_df_outdated, forced):
    zero_module_path = 'tests.zero_transform.TestZeroTransformStep'
    zero_last_modified_ts = 1.0
    drop_module_path = 'tests.drop_cols_transform.TestDropColsTransformStep'
    drop_last_modified_ts = 2.0
    valid_ts = max(zero_last_modified_ts, drop_last_modified_ts) + 1.0
    if is_df_outdated:
        df_last_modified_ts = min(zero_last_modified_ts, drop_last_modified_ts) - 1.0
    else:
        df_last_modified_ts = valid_ts

    transformed_format = 'transformed_format'
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format='init_format')
    # do nothing when it comes to save config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
    # override config getter
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)

    source_permanent_steps = [
        DfTransformStepConfig(module_path=zero_module_path,
                              params={'zero_cols': ['a']}),
    ]
    permanent_steps = [
        DfTransformStepConfig(module_path=drop_module_path,
                              params={'cols_to_drop': ['b']}),
    ]

    source_id = 'source_transform_id'
    source_transform = DfTransformConfig(transform_id=source_id,
                                         df_format=transformed_format,
                                         in_memory_steps=[],
                                         permanent_steps=source_permanent_steps)

    transform = DfTransformConfig(transform_id='transform_id',
                                  source_id=source_id,
                                  df_format=transformed_format,
                                  in_memory_steps=[],
                                  permanent_steps=permanent_steps)

    mocker.patch.object(DfReader,
                        DfReader.df_exists.__name__,
                        lambda _, df_id, transform_id: True)
    mocker.patch.object(DfReader,
                        DfReader._df_last_modified_ts.__name__,
                        lambda _, df_id, transform_id: df_last_modified_ts if transform_id == source_id else valid_ts)

    trans_cache = TestDfCache()
    trans_load_mock = mocker.patch.object(trans_cache, trans_cache.load.__name__)
    trans_df = mocker.Mock()
    trans_load_mock.return_value = trans_df
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        transformed_format: trans_cache
    })
    transform_dict = transform.to_dict()[1]
    source_transform_dict = source_transform.to_dict()[1]
    transform_state = {
        transform.transform_id: transform_dict,
        source_transform.transform_id: source_transform_dict
    }
    transform_dict[TRANSFORM_STATE_SOURCE_KEY] = source_transform_dict
    mocker.patch.object(DfReader, DfReader._is_file_exists.__name__, lambda path: True)
    mocker.patch.object(reader, DfReader._transforms_state_dicts.__name__, lambda df_id: transform_state)
    # ignore states for this test
    mocker.patch.object(reader, reader._save_transforms_state_file.__name__)
    #
    reader.register_transform(df_id=df_id, df_config=config, transform=source_transform)
    reader.register_transform(df_id=df_id, df_config=config, transform=transform)

    def last_modified_date(file_path: str):
        if 'zero_transform' in file_path:
            return zero_last_modified_ts
        elif 'drop_cols_transform' in file_path:
            return drop_last_modified_ts
        else:
            raise ValueError('???')

    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, last_modified_date)

    if is_df_outdated and not forced:
        with pytest.raises(Exception):
            reader.read(df_id=df_id, transform=transform, forced=forced)
    else:
        res_trans_df = reader.read(df_id=df_id, transform=transform, forced=forced)
        assert res_trans_df == trans_df


def _df_dir_path(dir_path: str, df_id: str) -> str:
    return f'{dir_path}{df_id}'
