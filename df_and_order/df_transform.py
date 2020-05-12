from typing import List, Optional

from df_and_order.df_transform_step import DfTransformStepConfig


TRANSFORM_ID_KEY = 'transform_id'
TRANSFORM_IN_MEMORY_KEY = 'in_memory'
TRANSFORM_PERMANENT_KEY = 'permanent'


class DfTransformConfig:
    def __init__(self,
                 transform_id: str,
                 in_memory_steps: Optional[List[DfTransformStepConfig]] = None,
                 permanent_steps: Optional[List[DfTransformStepConfig]] = None):
        assert in_memory_steps or permanent_steps, "Provide at least one type of transformations"

        self._transform_id = transform_id
        self._in_memory_steps = in_memory_steps
        self._permanent_steps = permanent_steps

    @property
    def transform_id(self) -> str:
        return self._transform_id

    @property
    def in_memory_steps(self) -> Optional[List[DfTransformStepConfig]]:
        return self._in_memory_steps

    @property
    def permanent_steps(self) -> Optional[List[DfTransformStepConfig]]:
        return self._permanent_steps

    def to_dict(self):
        result = {}

        if self._in_memory_steps:
            in_memory_dicts = [config.to_dict() for config in self._in_memory_steps]
            result[TRANSFORM_IN_MEMORY_KEY] = in_memory_dicts

        if self._permanent_steps:
            permanent_dicts = [config.to_dict() for config in self._permanent_steps]
            result[TRANSFORM_PERMANENT_KEY] = permanent_dicts

        return self.transform_id, result

    @staticmethod
    def from_dict(transform_id: str,
                  transform_dict: dict):
        in_memory_transforms = DfTransformConfig._transform_configs_from(transform_dict=transform_dict,
                                                                         key=TRANSFORM_IN_MEMORY_KEY)
        permanent_transforms = DfTransformConfig._transform_configs_from(transform_dict=transform_dict,
                                                                         key=TRANSFORM_PERMANENT_KEY)

        result = DfTransformConfig(transform_id=transform_id,
                                   in_memory_steps=in_memory_transforms,
                                   permanent_steps=permanent_transforms)
        return result

    @staticmethod
    def _transform_configs_from(transform_dict: dict,
                                key: str) -> Optional[List[DfTransformStepConfig]]:
        transform_dicts = transform_dict.get(key)
        transforms = None
        if transform_dicts:
            transforms = [
                DfTransformStepConfig.from_dict(transform_dict=transform_dict)
                for transform_dict in transform_dicts
            ]

        return transforms
