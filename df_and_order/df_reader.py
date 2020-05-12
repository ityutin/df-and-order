import os
import pandas as pd
from typing import Optional, Dict, List

from df_and_order.df_transform import DfTransformConfig
from df_and_order.df_config import DfConfig
from df_and_order.df_cache import DfCache
from df_and_order.df_transform_step import DfTransformStepConfig, DfTransformStep


class DfReader:
    """
    Reads a dataframe from the disk using a config file.
    Helps you to organize and document your dataframes
    for easy access and reproducibility.

    Parameters
    ----------
    dir_path: str
        Absolute path where your dataframe is supposed to be saved.
    format_to_cache_map: dict
        To be able to work with various formats DfReader must know
        how to read&save them. Provide a map where a format extension is used as a key
        and DfCache instance as a value. See DfCache class documentation for the details.
    """
    def __init__(self,
                 dir_path: str,
                 format_to_cache_map: Dict[str, DfCache]):
        self._dir_path = dir_path
        self._format_to_cache_map = format_to_cache_map

    def df_exists(self,
                  df_id: str,
                  transform_id: Optional[str] = None) -> bool:
        """
        Checks whether a dataframe file exists at the provided path.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        transform_id: optional str
            If you want to check whether a transformed version of your dataframe
            is persisted on the disk you may pass its unique identifier that
            matches the one in the config file.

        Returns
        -------
        True if the dataframe exists, False otherwise
        """
        df_dir_path = self._df_dir_path(df_id=df_id)
        if not DfConfig.config_exists(dir_path=df_dir_path):
            return False

        df_config = self._get_config(df_id=df_id)
        df_path = self._df_path(df_config=df_config,
                                df_id=df_id,
                                transform_id=transform_id)
        return os.path.exists(path=df_path)

    def create_df_config(self,
                         df_id: str,
                         initial_df_format: str,
                         transformed_df_format: str,
                         metadata: Optional[dict] = None,
                         transform: Optional[DfTransformConfig] = None):
        """
        Just a forwarding to DfConfig method, see docs in DfConfig.
        """
        DfConfig.create_config(dir_path=self._df_dir_path(df_id=df_id),
                               df_id=df_id,
                               initial_df_format=initial_df_format,
                               transformed_df_format=transformed_df_format,
                               metadata=metadata,
                               transform=transform)

    def register_transform(self,
                           df_id: str,
                           transform: DfTransformConfig):
        """
        Forms a filename for the given dataframe and adds a new transform to the config file if possible.
        In general it's just a forwarding to DfConfig method, see docs in DfConfig.
        """
        df_config = self._get_config(df_id=df_id)
        filename = self.df_filename(df_config=df_config,
                                    df_id=df_id,
                                    transform_id=transform.transform_id)
        df_config.register_transform(transform=transform,
                                     filename=filename)

    def read(self,
             df_id: str,
             transform_id: Optional[str] = None,
             transform: Optional[DfTransformConfig] = None) -> pd.DataFrame:
        """
        Reads a dataframe from the disk. If you want a transformed version of your dataframe,
        but it's still not persisted, it first creates it and then reads it into memory.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        transform_id: optional str
            Unique identifier of the desired transform.
        transform: optional DfTransformConfig
            Object that describes all the transformation steps required.

        Returns
        -------

        """
        if transform_id and transform:
            raise AttributeError('Provide either transform_id or transform_config')

        if transform:
            transform_id = transform.transform_id

        df_config = self._get_config(df_id=df_id)

        if transform_id:
            if not transform:
                transform = df_config.transforms_by(transform_id=transform_id)

            return self._read_transformed(df_id=df_id,
                                          transform=transform,
                                          df_config=df_config)
        else:
            return self._read_initial(df_id=df_id,
                                      df_config=df_config)

    def _read_initial(self,
                      df_id: str,
                      df_config: DfConfig) -> pd.DataFrame:
        """
        Reads the original dataframe from the disk
        """
        return self._read_df(df_id=df_id,
                             df_format=df_config.initial_df_format,
                             df_config=df_config)

    def _read_transformed(self,
                          df_id: str,
                          transform: DfTransformConfig,
                          df_config: DfConfig) -> pd.DataFrame:
        """
        Reads the transformed dataframe from the disk or creates it if needed.
        """
        transform_id = transform.transform_id
        transformed_df_exists = self.df_exists(df_id=df_id,
                                               transform_id=transform_id)
        df_format = df_config.transformed_df_format
        if transformed_df_exists:
            df = self._read_df(df_config=df_config,
                               df_id=df_id,
                               df_format=df_format,
                               transform_id=transform_id)

            if transform.in_memory_steps:
                df = DfReader._apply_transform_steps(df=df,
                                                     steps=transform.in_memory_steps)

            return df

        self.register_transform(df_id=df_id,
                                transform=transform)

        df = self._read_initial(df_id=df_id,
                                df_config=df_config)

        if transform.in_memory_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.in_memory_steps)

        if transform.permanent_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.permanent_steps)

            df_path = self._df_path(df_config=df_config,
                                    df_id=df_id,
                                    transform_id=transform_id)
            df_cache = self._get_df_cache(df_format=df_format)
            df_cache.save(df=df, path=df_path)

        return df

    @staticmethod
    def _apply_transform_steps(df: pd.DataFrame,
                               steps: List[DfTransformStepConfig]) -> pd.DataFrame:
        """
        Applies all the steps for a transformation on the given dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            Initial dataframe to perform transformations on.
        steps: list of DfTransformStepConfig
            List of objects that represent a step of the whole transformation.

        Returns
        -------
        pd.DataFrame, fully transformed initial dataframe
        """
        for step in steps:
            transform = DfTransformStep.build_transform(config=step)
            df = transform.transform(df=df)

        return df

    def _read_df(self,
                 df_id: str,
                 df_format: str,
                 df_config: DfConfig,
                 transform_id: Optional[str] = None) -> pd.DataFrame:
        """
        General method for reading a dataframe from the disk.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        df_format: str
            Format in which dataframe was saved.
        df_config: DfConfig
            Interface for working with a serialized dataframe config file.
        transform_id: optional str
            Unique identifier of the desired transform.

        Returns
        -------
        pd.DataFrame, the requested dataframe
        """
        df_path = self._df_path(df_config=df_config,
                                df_id=df_id,
                                transform_id=transform_id)

        df_cache = self._get_df_cache(df_format=df_format)
        df = df_cache.load(path=df_path)

        return df

    def _df_path(self,
                 df_config: DfConfig,
                 df_id: str,
                 transform_id: Optional[str] = None) -> str:
        """
        Forms a path to the dataframe.

        Parameters
        ----------
        df_config: DfConfig
            Interface for working with a serialized dataframe config file.
        df_id: str
            Unique identifier of your dataframe.
        transform_id: optional str
            Unique identifier of the desired transform.

        Returns
        -------
        str, absolute path to the dataframe
        """
        filename = DfReader.df_filename(df_config=df_config,
                                        df_id=df_id,
                                        transform_id=transform_id)
        result = self._df_dir_path(df_id=df_id, filename=filename)

        return result

    @staticmethod
    def df_filename(df_config: DfConfig,
                    df_id: str,
                    transform_id: Optional[str] = None):
        """
        Forms a filename for the dataframe

        Parameters
        ----------
        df_config: DfConfig
            Interface for working with a serialized dataframe config file.
        df_id: str
            Unique identifier of your dataframe.
        transform_id: optional str
            Unique identifier of the desired transform.

        Returns
        -------
        str, a filename for the dataframe.
        """
        if transform_id:
            return f'{transform_id}_{df_id}.{df_config.transformed_df_format}'

        return f'{df_id}.{df_config.initial_df_format}'

    def _df_dir_path(self, df_id: str, filename: Optional[str] = None) -> str:
        """
        Forms an absolute file for a directory where you can find your dataframe or any other file.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        filename: optional str
            If you want to get a path for some particular file, provide its filename.

        Returns
        -------
        str, absolute path to the desired item
        """
        path = self._df_dir_path_for(dir_path=self._dir_path,
                                     df_id=df_id)

        if filename:
            path = os.path.join(path, filename)

        return path

    @staticmethod
    def _df_dir_path_for(dir_path: str,
                         df_id: str) -> str:
        return os.path.join(dir_path, df_id)

    def _get_config(self, df_id: str) -> DfConfig:
        """
        Gets config object for the given dataframe.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.

        Returns
        -------
        DfConfig instance.
        """
        result = self._get_config_for(dir_path=self._df_dir_path(df_id=df_id),
                                      df_id=df_id)
        return result

    @staticmethod
    def _get_config_for(dir_path: str, df_id: str) -> DfConfig:
        """
        Gets config object for the given dataframe at the given path.

        Parameters
        ----------
        dir_path: str
            Absolute path to the dir when the dataframe is located.
        df_id: str
            Unique identifier of your dataframe.

        Returns
        -------

        """
        result = DfConfig(df_id=df_id,
                          dir_path=dir_path)
        return result

    def _get_df_cache(self, df_format: str) -> DfCache:
        """
        Gives proper DfCache subclass for the given data format

        Parameters
        ----------
        df_format: str
            Format extension you want to use to save/load a dataframe.

        Returns
        -------
        DfCache instance.
        """
        df_cache = self._format_to_cache_map.get(df_format)
        if not df_cache:
            raise ValueError(f'Unknown df_format, df_cache was not provided: {df_format}')
        return df_cache
