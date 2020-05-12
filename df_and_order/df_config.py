from typing import Optional
import os
import yaml

from df_and_order.df_transform import DfTransformConfig

DF_ID_KEY = 'df_id'
DF_INITIAL_FORMAT_KEY = 'initial_df_format'
DF_TRANSFORMED_FORMAT_KEY = 'transformed_df_format'
METADATA_KEY = 'metadata'
TRANSFORMS_KEY = 'transforms'


class DfConfig:
    """
    Interface for working with a serialized dataframe config file.

    Init parameters
    ----------
    df_id: str
        Unique identifier of your dataframe.
    dir_path: str
        Location of a folder where config file should be read from.
    """
    def __init__(self, df_id: str, dir_path: str):
        self._dir_path = dir_path
        with open(self._config_path()) as config_file:
            config_dict = yaml.safe_load(config_file)

        df_check = config_dict[DF_ID_KEY] == df_id
        assert df_check, "config_dict doesn't belong to requested dataframe"

        self._config_dict = config_dict

    @staticmethod
    def config_exists(dir_path: str) -> bool:
        """
        Checks whether a dataframe config file exists at the provided path.

        Parameters
        ----------
        dir_path: str
            Location of a folder where config file should be read from.

        Returns
        -------
        True if config file exists, False otherwise
        """
        path_to_check = DfConfig._config_path_for(dir_path=dir_path)
        return os.path.exists(path_to_check)

    def _config_path(self) -> str:
        """
        Gets a path to the config using the given dir_path
        """
        return self._config_path_for(dir_path=self._dir_path)

    @staticmethod
    def _config_path_for(dir_path: str) -> str:
        """
        Gets a path to the config using the given dir_path
        """
        return DfConfig._path_for_file(dir_path=dir_path, filename='df_config.yaml')

    @staticmethod
    def _path_for_file(dir_path: str, filename: str) -> str:
        return os.path.join(dir_path, filename)

    @staticmethod
    def create_config(dir_path: str,
                      df_id: str,
                      initial_df_format: str,
                      transformed_df_format: str,
                      metadata: Optional[dict] = None,
                      transform: Optional[DfTransformConfig] = None):
        """
        Creates config file at the given location.

        Parameters
        ----------
        dir_path: str
            Location of a folder where config file should be saved to
        df_id: str
            Unique identifier of your dataframe.
        initial_df_format: str
            Format extension for an initial dataframe before any transformation.
        transformed_df_format: str
            # TODO: let a transformation decide in which format to save a dataframe
            Format extension for a transformed dataframe.
        metadata: optional dict
            Any information you want to store in the config file as a reference or for future use.
        transform: DfTransformConfig
            Object that describes all the transformation steps required.

        Returns
        -------
        Nothing, but creates the config in the filesystem.
        """
        assert not DfConfig.config_exists(dir_path=dir_path), f"Config for df {df_id} already exists"

        config_path = DfConfig._config_path_for(dir_path=dir_path)
        config_dict = {
            DF_ID_KEY: df_id,
            DF_INITIAL_FORMAT_KEY: initial_df_format,
            DF_TRANSFORMED_FORMAT_KEY: transformed_df_format,
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

    @property
    def initial_df_format(self) -> str:
        return self._config_dict[DF_INITIAL_FORMAT_KEY]

    @property
    def transformed_df_format(self) -> str:
        return self._config_dict[DF_TRANSFORMED_FORMAT_KEY]

    def transforms_by(self, transform_id: str) -> DfTransformConfig:
        """
        Gets the transform representation from the config file.

        Parameters
        ----------
        transform_id: str
            Unique identifier of a transform

        Returns
        -------
        DfTransformConfig that describes all the transformation steps.
        """
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
        """
        Adds a new transform to the config file if possible.

        Parameters
        ----------
        transform: DfTransformConfig
            Object that describes all the transformation steps required.
        filename: str
            What a filename for the transformation would be to look for already existing one.

        Returns
        -------
        Nothing, but updates the config in the filesystem.
        """
        transform_id, transform_dict = transform.to_dict()
        maybe_transforms = self._config_dict.get(TRANSFORMS_KEY, {})
        if maybe_transforms and maybe_transforms.get(transform_id):
            full_filename = f'{filename}.{self._config_dict[DF_TRANSFORMED_FORMAT_KEY]}'
            already_cached = os.path.exists(self._path_for_file(dir_path=self._dir_path, filename=full_filename))
            assert not already_cached, f"Result of the transform {transform_id} already exists."

        maybe_transforms[transform_id] = transform_dict
        self._config_dict[TRANSFORMS_KEY] = maybe_transforms
        self._save()

    def _save(self):
        """
        Serializes config to the disk
        """
        self._save_at_path(config_dict=self._config_dict,
                           config_path=self._config_path())

    @staticmethod
    def _save_at_path(config_dict: dict, config_path: str):
        with open(config_path, 'w') as config_file:
            yaml.dump(config_dict, config_file)
