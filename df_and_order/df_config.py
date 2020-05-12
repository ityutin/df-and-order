from typing import Optional, List
import os
import yaml

from df_and_order.df_transform_step import DfTransformStepConfig
from df_and_order.df_transform import DfTransformConfig

DATASET_ID_KEY = 'dataset_id'
DATASET_INITIAL_FORMAT_KEY = 'initial_dataset_format'
DATASET_TRANSFORMED_FORMAT_KEY = 'transformed_dataset_format'
METADATA_KEY = 'metadata'
TRANSFORMS_KEY = 'transforms'


class DfConfig:
    def __init__(self, dataset_id: str, dir_path: str):
        self._dir_path = dir_path
        with open(self._config_path()) as config_file:
            config_dict = yaml.safe_load(config_file)

        dataset_check = config_dict[DATASET_ID_KEY] == dataset_id
        assert dataset_check, "config_dict doesn't belong to requested dataset"

        self._config_dict = config_dict

    @staticmethod
    def config_exists(dir_path: str) -> bool:
        path_to_check = DfConfig._config_path_for(dir_path=dir_path)
        return os.path.exists(path_to_check)

    def _config_path(self) -> str:
        return self._config_path_for(dir_path=self._dir_path)

    @staticmethod
    def _config_path_for(dir_path: str) -> str:
        return DfConfig._path_for_file(dir_path=dir_path, filename='dataset_config.yaml')

    @staticmethod
    def _path_for_file(dir_path: str, filename: str) -> str:
        return os.path.join(dir_path, filename)

    @staticmethod
    def create_config(dir_path: str,
                      dataset_id: str,
                      initial_dataset_format: str,
                      transformed_dataset_format: str,
                      metadata: Optional[dict] = None,
                      transform: Optional[DfTransformConfig] = None):
        assert not DfConfig.config_exists(dir_path=dir_path), f"Config for df {dataset_id} already exists"

        config_path = DfConfig._config_path_for(dir_path=dir_path)
        config_dict = {
            DATASET_ID_KEY: dataset_id,
            DATASET_INITIAL_FORMAT_KEY: initial_dataset_format,
            DATASET_TRANSFORMED_FORMAT_KEY: transformed_dataset_format,
        }
        if metadata:
            config_dict[METADATA_KEY] = metadata

        if transform:
            transform_id, transform_dict = transform.to_dict()
            config_dict[TRANSFORMS_KEY] = {
                transform_id: transform_dict
            }

        DfConfig._save_at_path(config_dict=config_dict,
                               config_path=config_path)

    @staticmethod
    def _add_transform_configs(config_dict: dict,
                               key: str,
                               transforms: List[DfTransformStepConfig]):
        transform_dicts = [config.to_dict() for config in transforms]
        transforms_list = config_dict.get(key, [])
        transforms_list += transform_dicts
        config_dict[key] = transforms_list
        return config_dict

    @property
    def initial_dataset_format(self) -> str:
        return self._config_dict[DATASET_INITIAL_FORMAT_KEY]

    @property
    def transformed_dataset_format(self) -> str:
        return self._config_dict[DATASET_TRANSFORMED_FORMAT_KEY]

    def transforms_by(self, transform_id: str) -> DfTransformConfig:
        maybe_transforms_dict = self._config_dict.get(TRANSFORMS_KEY)
        assert maybe_transforms_dict, "Config contains no transforms"
        maybe_transform_dict = maybe_transforms_dict.get(transform_id)
        assert maybe_transform_dict, f"Requested transform {transform_id} was not found"
        result = DfTransformConfig.from_dict(transform_id=transform_id,
                                             transform_dict=maybe_transform_dict)

        return result

    def register_transform(self,
                           transform: DfTransformConfig,
                           filename: str):
        transform_id, transform_dict = transform.to_dict()
        maybe_transforms = self._config_dict.get(TRANSFORMS_KEY, {})
        if maybe_transforms and maybe_transforms.get(transform_id):
            full_filename = f'{filename}.{self._config_dict[DATASET_TRANSFORMED_FORMAT_KEY]}'
            already_cached = os.path.exists(self._path_for_file(dir_path=self._dir_path, filename=full_filename))
            assert not already_cached, f"Result of the transform {transform_id} already exists."

        maybe_transforms[transform_id] = transform_dict
        self._config_dict[TRANSFORMS_KEY] = maybe_transforms
        self._save()

    def _save(self):
        self._save_at_path(config_dict=self._config_dict,
                           config_path=self._config_path())

    @staticmethod
    def _save_at_path(config_dict: dict, config_path: str):
        with open(config_path, 'w') as config_file:
            yaml.dump(config_dict, config_file)
