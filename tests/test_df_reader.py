import pytest
import pandas as pd
import numpy as np
from df_and_order.df_config import DfConfig, DF_ID_KEY, DF_INITIAL_FORMAT_KEY, DF_TRANSFORMED_FORMAT_KEY
from df_and_order.df_reader import DfReader
from df_and_order.df_transform import DfTransformConfig
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


@pytest.fixture()
def initial_df():
    result = pd.DataFrame({
        'a': ['2020-02-01', '2020-02-02', '2020-02-03'],
        'b': [1, 2, 3],
        'c': ['2020-01-01', '2020-01-02', '2020-01-03']
    })
    return result


def build_config(mocker, df_id, test_path, initial_format, transformed_format):
    config_dict = {
        DF_ID_KEY: df_id,
        DF_INITIAL_FORMAT_KEY: initial_format,
        DF_TRANSFORMED_FORMAT_KEY: transformed_format
    }

    result = _get_config(mocker=mocker, df_id=df_id, dir_path=test_path, config_dict=config_dict)
    return result


@pytest.mark.parametrize("transform_id", ['trans_1', None], ids=['trans', 'no_trans'])
@pytest.mark.parametrize("valid_case", [True, False], ids=['valid', 'not_valid'])
def test_df_exists(mocker, df_id, test_path, transform_id, valid_case):
    mocker.patch.object(DfConfig, DfConfig.config_exists.__name__, lambda dir_path: True)
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format='init_format',
                          transformed_format='trans_format')
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)

    if transform_id:
        filename = f'{transform_id}_{df_id}.{config.transformed_df_format}'
    else:
        filename = f'{df_id}.{config.initial_df_format}'

    if not valid_case:
        filename += 'garbage'

    test_file_path = f'{test_path}{df_id}/{filename}'
    mocker.patch.object(DfReader, DfReader._is_file_exists.__name__, lambda path: path == test_file_path)

    reader = DfReader(dir_path=test_path, format_to_cache_map={})
    if valid_case:
        assert reader.df_exists(df_id=df_id, transform_id=transform_id)
    else:
        assert not reader.df_exists(df_id=df_id, transform_id=transform_id)


def test_create_df_config(mocker, df_id, test_path):
    reader = DfReader(dir_path=test_path, format_to_cache_map={})

    initial_format = mocker.Mock()
    transformed_df_format = mocker.Mock()
    metadata = mocker.Mock()
    transform = mocker.Mock()

    config_create_mock = mocker.patch.object(DfConfig, DfConfig.create_config.__name__)

    reader.create_df_config(df_id=df_id,
                            initial_df_format=initial_format,
                            transformed_df_format=transformed_df_format,
                            metadata=metadata,
                            transform=transform)

    dir_path = _df_dir_path(dir_path=test_path, df_id=df_id)
    config_create_mock.assert_called_with(dir_path=dir_path,
                                          df_id=df_id,
                                          initial_df_format=initial_format,
                                          transformed_df_format=transformed_df_format,
                                          metadata=metadata,
                                          transform=transform)


def test_register_transform(mocker, df_id, test_path):
    config = build_config(mocker=mocker,
                          df_id=df_id,
                          test_path=test_path,
                          initial_format='init_format',
                          transformed_format='trans_format')
    mocker.patch.object(DfReader, DfReader._get_config.__name__, lambda _, df_id: config)
    register_transform_mock = mocker.patch.object(DfConfig, DfConfig.register_transform.__name__)

    transform = mocker.Mock()
    transform.transform_id = 'trans_id'
    filename = f'{transform.transform_id}_{df_id}.{config.transformed_df_format}'

    reader = DfReader(dir_path=test_path, format_to_cache_map={})
    reader.register_transform(df_id=df_id,
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
                          initial_format=init_format,
                          transformed_format='trans_format')
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
                          initial_format=init_format,
                          transformed_format=transformed_format)
    # do nothing when it comes to save the config
    mocker.patch.object(DfConfig, DfConfig._save.__name__)
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
        df_path = f'{df_dir_path}/{transform.transform_id}_{df_id}.{config.transformed_df_format}'
        trans_save_mock.assert_called_with(df=transformed_df,
                                           path=df_path)


@pytest.mark.parametrize("is_df_outdated", [True, False], ids=['outdated', 'not_outdated'])
def test_read_transformed_check_ts(mocker, df_id, test_path, initial_df, is_df_outdated):
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
                          initial_format='init_format',
                          transformed_format=transformed_format)
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
                                  in_memory_steps=[],
                                  permanent_steps=permanent_steps)

    trans_cache = TestDfCache()
    trans_load_mock = mocker.patch.object(trans_cache, trans_cache.load.__name__)
    trans_df = mocker.Mock()
    trans_load_mock.return_value = trans_df
    reader = DfReader(dir_path=test_path, format_to_cache_map={
        transformed_format: trans_cache
    })

    def last_modified_date(file_path: str):
        if 'zero_transform' in file_path:
            return zero_last_modified_ts
        elif 'drop_cols_transform' in file_path:
            return drop_last_modified_ts
        else:
            raise ValueError('???')

    mocker.patch.object(FileInspector, FileInspector.last_modified_date.__name__, last_modified_date)

    if is_df_outdated:
        with pytest.raises(Exception):
            reader.read(df_id=df_id, transform=transform)
    else:
        res_trans_df = reader.read(df_id=df_id, transform=transform)
        assert res_trans_df == trans_df


def _df_dir_path(dir_path: str, df_id: str) -> str:
    return f'{dir_path}{df_id}'
