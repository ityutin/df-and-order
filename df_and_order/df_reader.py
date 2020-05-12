import os
import pandas as pd
from typing import Optional, Dict, List

from df_and_order.df_transform import DfTransformConfig
from df_and_order.df_config import DfConfig
from df_and_order.df_cache import DfCache
from df_and_order.df_transform_step import DfTransformStepConfig, DfTransformStep


class DfReader:
    def __init__(self,
                 dir_path: str,
                 format_to_cache_map: Dict[str, DfCache]):
        self._dir_path = dir_path
        self._format_to_cache_map = format_to_cache_map

    def dataset_exists(self,
                       dataset_id: str,
                       transform_id: Optional[str] = None) -> bool:
        dataset_dir_path = self._dataset_dir_path(dataset_id=dataset_id)
        if not DfConfig.config_exists(dir_path=dataset_dir_path):
            return False

        dataset_config = self._get_config(dataset_id=dataset_id)
        df_path = self._df_path(dataset_config=dataset_config,
                                dataset_id=dataset_id,
                                transform_id=transform_id)
        return os.path.exists(path=df_path)

    def create_dataset_config(self,
                              dataset_id: str,
                              initial_dataset_format: str,
                              transformed_dataset_format: str,
                              metadata: Optional[dict] = None,
                              transform: Optional[DfTransformConfig] = None):
        DfConfig.create_config(dir_path=self._dataset_dir_path(dataset_id=dataset_id),
                               dataset_id=dataset_id,
                               initial_dataset_format=initial_dataset_format,
                               transformed_dataset_format=transformed_dataset_format,
                               metadata=metadata,
                               transform=transform)

    def register_transform(self,
                           dataset_id: str,
                           transform: DfTransformConfig):
        dataset_config = self._get_config(dataset_id=dataset_id)
        filename = self.dataset_filename(dataset_config=dataset_config,
                                         dataset_id=dataset_id,
                                         transform_id=transform.transform_id)
        dataset_config.register_transform(transform=transform,
                                          filename=filename)

    def read(self,
             dataset_id: str,
             transform_id: Optional[str] = None,
             transform: Optional[DfTransformConfig] = None) -> pd.DataFrame:
        if transform_id and transform:
            raise AttributeError('Provide either transform_id or transform_config')

        if transform:
            transform_id = transform.transform_id

        dataset_config = self._get_config(dataset_id=dataset_id)

        if transform_id:
            if not transform:
                transform = dataset_config.transforms_by(transform_id=transform_id)

            return self._read_transformed(dataset_id=dataset_id,
                                          transform=transform,
                                          dataset_config=dataset_config)
        else:
            return self._read_initial(dataset_id=dataset_id,
                                      dataset_config=dataset_config)

    def _read_initial(self,
                      dataset_id: str,
                      dataset_config: DfConfig) -> pd.DataFrame:
        return self._read_df(dataset_id=dataset_id,
                             dataset_format=dataset_config.initial_dataset_format,
                             dataset_config=dataset_config)

    def _read_transformed(self,
                          dataset_id: str,
                          transform: DfTransformConfig,
                          dataset_config: DfConfig) -> pd.DataFrame:
        transform_id = transform.transform_id
        transformed_dataset_exists = self.dataset_exists(dataset_id=dataset_id,
                                                         transform_id=transform_id)
        dataset_format = dataset_config.transformed_dataset_format
        if transformed_dataset_exists:
            df = self._read_df(dataset_config=dataset_config,
                               dataset_id=dataset_id,
                               dataset_format=dataset_format,
                               transform_id=transform_id)

            if transform.in_memory_steps:
                df = DfReader._apply_transform_steps(df=df,
                                                     steps=transform.in_memory_steps)

            return df

        self.register_transform(dataset_id=dataset_id,
                                transform=transform)

        df = self._read_initial(dataset_id=dataset_id,
                                dataset_config=dataset_config)

        if transform.in_memory_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.in_memory_steps)

        if transform.permanent_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.permanent_steps)

            df_path = self._df_path(dataset_config=dataset_config,
                                    dataset_id=dataset_id,
                                    transform_id=transform_id)
            df_cache = self._get_df_cache(dataset_format=dataset_format)
            df_cache.save(df=df, path=df_path)

        return df

    @staticmethod
    def _apply_transform_steps(df: pd.DataFrame,
                               steps: List[DfTransformStepConfig]) -> pd.DataFrame:
        for step in steps:
            transform = DfTransformStep.build_transform(config=step)
            df = transform.transform(df=df)

        return df

    def _read_df(self,
                 dataset_id: str,
                 dataset_format: str,
                 dataset_config: DfConfig,
                 transform_id: Optional[str] = None):
        df_path = self._df_path(dataset_config=dataset_config,
                                dataset_id=dataset_id,
                                transform_id=transform_id)

        df_cache = self._get_df_cache(dataset_format=dataset_format)
        df = df_cache.load(path=df_path)

        return df

    def _df_path(self,
                 dataset_config: DfConfig,
                 dataset_id: str,
                 transform_id: Optional[str] = None):
        filename = DfReader.dataset_filename(dataset_config=dataset_config,
                                             dataset_id=dataset_id,
                                             transform_id=transform_id)
        result = self._dataset_dir_path(dataset_id=dataset_id, filename=filename)

        return result

    @staticmethod
    def dataset_filename(dataset_config: DfConfig,
                         dataset_id: str,
                         transform_id: Optional[str] = None):
        if transform_id:
            return f'{transform_id}_{dataset_id}.{dataset_config.transformed_dataset_format}'

        return f'{dataset_id}.{dataset_config.initial_dataset_format}'

    def _dataset_dir_path(self, dataset_id: str, filename: Optional[str] = None) -> str:
        path = self._dataset_dir_path_for(dir_path=self._dir_path,
                                          dataset_id=dataset_id)

        if filename:
            path = os.path.join(path, filename)

        return path

    @staticmethod
    def _dataset_dir_path_for(dir_path: str,
                              dataset_id: str) -> str:
        return os.path.join(dir_path, dataset_id)

    def _get_config(self, dataset_id: str) -> DfConfig:
        result = self._get_config_for(dir_path=self._dataset_dir_path(dataset_id=dataset_id), dataset_id=dataset_id)
        return result

    @staticmethod
    def _get_config_for(dir_path: str, dataset_id) -> DfConfig:
        result = DfConfig(dataset_id=dataset_id,
                          dir_path=dir_path)
        return result

    def _get_df_cache(self, dataset_format: str) -> DfCache:
        df_cache = self._format_to_cache_map.get(dataset_format)
        if not df_cache:
            raise ValueError(f'Unknown dataset_format, df_cache was not provided: {dataset_format}')
        return df_cache
