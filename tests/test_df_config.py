import pytest
import copy
from df_and_order.df_config import DfConfig, DF_ID_KEY, CONFIG_FILENAME, DF_INITIAL_FORMAT_KEY, METADATA_KEY, TRANSFORMS_KEY
from df_and_order.df_transform import DfTransformConfig
from df_and_order.df_transform_step import DfTransformStepConfig


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
def transform(transformed_format):
    transform = DfTransformConfig(transform_id='transform_id',
                                  df_format=transformed_format,
                                  in_memory_steps=[
                                    DfTransformStepConfig(module_path='123', params={'a': 1, 'b': 'b'})
                                  ],
                                  permanent_steps=[
                                    DfTransformStepConfig(module_path='456', params={'c': 2, 'd': 1.2})
                                  ])
    return transform


def test_df_id_match(mocker, df_id, test_path):
    valid_df_id = df_id
    mocker.patch.object(DfConfig, DfConfig._read_config.__name__, lambda _, path: {DF_ID_KEY: valid_df_id})
    DfConfig(df_id=df_id, dir_path=test_path)


def test_df_id_mismatch(mocker, df_id, test_path):
    wrong_df_id = 'wrong_id'
    mocker.patch.object(DfConfig, DfConfig._read_config.__name__, lambda _, path: {DF_ID_KEY: wrong_df_id})
    with pytest.raises(Exception):
        DfConfig(df_id=df_id, dir_path=test_path)


def test_config_exists(mocker, df_id, test_path):
    config_path = test_path + CONFIG_FILENAME
    mocker.patch.object(DfConfig, DfConfig._is_file_exists.__name__, lambda path: path == config_path)
    assert DfConfig.config_exists(dir_path=test_path)


def test_create_config_when_existing(mocker, df_id, test_path):
    mocker.patch.object(DfConfig, DfConfig.config_exists.__name__, lambda dir_path: True)
    save_mock = mocker.patch.object(DfConfig, DfConfig._save_at_path.__name__)

    with pytest.raises(Exception):
        DfConfig.create_config(dir_path=test_path,
                               df_id=df_id,
                               initial_df_format='')

    save_mock.assert_not_called()


@pytest.mark.parametrize("metadata", [{'meta': 'shmeta'}, None], ids=['meta', 'no_meta'])
@pytest.mark.parametrize("use_transform", [True, False], ids=['transform', 'no_transform'])
def test_create_config(mocker, df_id, test_path, transform, metadata, use_transform):
    initial_df_format = 'init_format'

    save_mock = mocker.patch.object(DfConfig, DfConfig._save_at_path.__name__)

    DfConfig.create_config(dir_path=test_path,
                           df_id=df_id,
                           initial_df_format=initial_df_format,
                           metadata=metadata,
                           transform=transform if use_transform else None)

    config_dict = {
        DF_ID_KEY: df_id,
        DF_INITIAL_FORMAT_KEY: initial_df_format,
    }

    if metadata:
        config_dict[METADATA_KEY] = metadata

    if use_transform:
        transform_id, transform_dict = transform.to_dict()
        config_dict[TRANSFORMS_KEY] = {
            transform_id: transform_dict
        }

    save_mock.assert_called_with(config_dict=config_dict,
                                 config_path=test_path + CONFIG_FILENAME)


def test_properties(mocker, df_id, test_path):
    initial_df_format = 'init_format'
    config_dict = {
        DF_ID_KEY: df_id,
        DF_INITIAL_FORMAT_KEY: initial_df_format,
    }
    config = _get_config(mocker=mocker,
                         df_id=df_id,
                         dir_path=test_path,
                         config_dict=config_dict)

    assert config.initial_df_format == initial_df_format
    assert config.df_id == df_id


def test_transform_by_no_transforms(mocker, df_id, test_path):
    config_dict = {
        DF_ID_KEY: df_id,
    }
    config = _get_config(mocker=mocker,
                         df_id=df_id,
                         dir_path=test_path,
                         config_dict=config_dict)

    with pytest.raises(Exception):
        config.transforms_by(transform_id='whatever')


def test_transform_by_no_transform(mocker, df_id, test_path):
    config_dict = {
        DF_ID_KEY: df_id,
        TRANSFORMS_KEY: {
            'some_transform_id': {}
        }
    }
    config = _get_config(mocker=mocker,
                         df_id=df_id,
                         dir_path=test_path,
                         config_dict=config_dict)

    with pytest.raises(Exception):
        config.transforms_by(transform_id='whatever')


def test_transform_by(mocker, df_id, test_path, transform):
    transform_id, transform_dict = transform.to_dict()

    config_dict = {
        DF_ID_KEY: df_id,
        TRANSFORMS_KEY: {
            transform_id: transform_dict
        }
    }
    config = _get_config(mocker=mocker,
                         df_id=df_id,
                         dir_path=test_path,
                         config_dict=config_dict)

    transform_result = config.transforms_by(transform_id=transform_id)
    _, transform_result_dict = transform_result.to_dict()

    assert transform_dict == transform_result_dict


def test_register_transform_already_cached(mocker, df_id, test_path, transform, transformed_format):
    initial_df_format = 'init_format'
    transform_id, transform_dict = transform.to_dict()
    config_dict = {
        DF_ID_KEY: df_id,
        DF_INITIAL_FORMAT_KEY: initial_df_format,
        TRANSFORMS_KEY: {
            transform_id: transform_dict
        }
    }
    transfom_filename = f'{transform_id}_{df_id}'
    config = _get_config(mocker=mocker,
                         df_id=df_id,
                         dir_path=test_path,
                         config_dict=config_dict)

    test_path_to_cached_file = test_path + transfom_filename + f'.{transformed_format}'
    is_file_exists_mock = mocker.patch.object(DfConfig, DfConfig._is_file_exists.__name__)
    is_file_exists_mock.return_value = True

    with pytest.raises(Exception):
        config.register_transform(transform=transform,
                                  filename=transfom_filename)

    is_file_exists_mock.assert_called_with(path=test_path_to_cached_file)


def test_register_transform(mocker, df_id, test_path, transform, transformed_format):
    save_mock = mocker.patch.object(DfConfig, DfConfig._save_at_path.__name__)
    initial_df_format = 'init_format'
    transform_id, transform_dict = transform.to_dict()
    config_dict = {
        DF_ID_KEY: df_id,
        DF_INITIAL_FORMAT_KEY: initial_df_format,
        TRANSFORMS_KEY: {
            transform_id: transform_dict
        }
    }
    config = _get_config(mocker=mocker,
                         df_id=df_id,
                         dir_path=test_path,
                         config_dict=copy.deepcopy(config_dict))

    updated_transform = DfTransformConfig(transform_id=transform_id,
                                          df_format=transformed_format,
                                          in_memory_steps=[
                                              DfTransformStepConfig(module_path='razdva', params={'asd': 123})
                                          ])

    config.register_transform(transform=updated_transform, filename='whatever')

    _, updated_transform_config = updated_transform.to_dict()

    config_dict[TRANSFORMS_KEY] = {
        transform_id: updated_transform_config
    }
    save_mock.assert_called_with(config_dict=config_dict,
                                 config_path=test_path + CONFIG_FILENAME)


def _get_config(mocker, df_id, dir_path, config_dict: dict):
    mocker.patch.object(DfConfig, DfConfig._read_config.__name__, lambda _, path: config_dict)
    config = DfConfig(df_id=df_id, dir_path=dir_path)
    return config
