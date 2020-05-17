import pandas as pd
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

from df_and_order.helpers import build_class_instance, get_file_path_from_module_path, FileInspector

TRANSFORM_STEP_MODULE_PATH_KEY = 'module_path'
TRANSFORM_STEP_PARAMS_KEY = 'params'


@dataclass
class DfTransformStepConfig:
    """
    Dataclass for storing module path of some DfTransformStep
    as well as its init parameters.
    """
    module_path: str
    params: dict

    def step_last_modified_ts(self) -> float:
        file_path = get_file_path_from_module_path(module_path=self.module_path)
        result = FileInspector.last_modified_date(file_path=file_path)
        return result

    @staticmethod
    def from_dict(step_dict: dict):
        module_path = step_dict[TRANSFORM_STEP_MODULE_PATH_KEY]
        params = step_dict.get(TRANSFORM_STEP_PARAMS_KEY) or {}
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
