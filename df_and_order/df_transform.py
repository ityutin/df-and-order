import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple

from df_and_order.helpers import build_class_instance


@dataclass
class DfTransformConfig:
    transform_id: str
    module_path: str
    needs_cache: bool
    params: dict

    @staticmethod
    def from_dict(transform_id: str,
                  transform_dict: dict):
        module_path = transform_dict[TRANSFORM_MODULE_PATH_KEY]
        needs_cache = transform_dict[TRANSFORM_NEED_CACHE_KEY]
        params = transform_dict[TRANSFORM_PARAMS_KEY]
        config = DfTransformConfig(transform_id=transform_id,
                                   module_path=module_path,
                                   needs_cache=needs_cache,
                                   params=params)
        return config

    def to_dict(self) -> Tuple[str, dict]:
        transform_dict = {
            TRANSFORM_MODULE_PATH_KEY: self.module_path,
            TRANSFORM_NEED_CACHE_KEY: self.needs_cache,
            TRANSFORM_PARAMS_KEY: self.params,
        }

        return self.transform_id, transform_dict
TRANSFORM_ID_KEY = 'transform_id'
TRANSFORM_MODULE_PATH_KEY = 'module_path'
TRANSFORM_PARAMS_KEY = 'params'
TRANSFORM_NEED_CACHE_KEY = 'needs_cache'


class DfTransform(ABC):
    def __init__(self, transform_id: str, needs_cache: bool):
        self._transform_id = transform_id
        self._needs_cache = needs_cache

    @staticmethod
    def build_transform(config: DfTransformConfig):
        params = {
            TRANSFORM_ID_KEY: config.transform_id,
            TRANSFORM_NEED_CACHE_KEY: config.needs_cache
        }

        params = {**params, **config.params}

        transform = build_class_instance(module_path=config.module_path,
                                         init_params=params)
        return transform

    @property
    def transform_id(self) -> str:
        return self._transform_id

    @property
    def needs_cache(self) -> bool:
        return self._needs_cache

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropColsTransform(DfTransform):
    def __init__(self, transform_id: str, needs_cache: bool, cols_to_drop: List[str]):
        super().__init__(transform_id=transform_id, needs_cache=needs_cache)

        self._cols_to_drop = cols_to_drop

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(self._cols_to_drop, axis=1)