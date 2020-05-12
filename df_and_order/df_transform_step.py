import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

from df_and_order.helpers import build_class_instance


TRANSFORM_STEP_MODULE_PATH_KEY = 'module_path'
TRANSFORM_STEP_PARAMS_KEY = 'params'
TRANSFORM_STEP_NEED_CACHE_KEY = 'needs_cache'


@dataclass
class DfTransformStepConfig:
    """
    Dataclass for storing module path of some DfTransformStep
    as well as its init parameters.
    """
    module_path: str
    params: dict

    @staticmethod
    def from_dict(transform_dict: dict):
        module_path = transform_dict[TRANSFORM_STEP_MODULE_PATH_KEY]
        params = transform_dict.get(TRANSFORM_STEP_PARAMS_KEY) or {}
        config = DfTransformStepConfig(module_path=module_path,
                                       params=params)
        return config

    def to_dict(self) -> dict:
        step_dict = {
            TRANSFORM_STEP_MODULE_PATH_KEY: self.module_path,
        }

        if len(self.params):
            step_dict[TRANSFORM_STEP_PARAMS_KEY] = self.params

        return step_dict


class DfTransformStep(ABC):
    """
    Encapsulates logic of some dataframe transformation.
    Every subclass must implement 'transform' method with
    custom logic.
    """
    @staticmethod
    def build_transform(config: DfTransformStepConfig):
        """
        Creates DfTransformStep instance out of config.

        Parameters
        ----------
        config: DfTransformStepConfig
            Contains information about DfTransformStep subclass
            and parameters to init it with.

        Returns
        -------
        DfTransformStep instance.
        """
        params = config.params

        transform = build_class_instance(module_path=config.module_path,
                                         init_params=params)
        return transform

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
