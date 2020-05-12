from typing import List, Optional, Tuple

from df_and_order.df_transform_step import DfTransformStepConfig


TRANSFORM_ID_KEY = 'transform_id'
TRANSFORM_IN_MEMORY_KEY = 'in_memory'
TRANSFORM_PERMANENT_KEY = 'permanent'


class DfTransformConfig:
    """
    Describes how a transformation should be performed.
    For any transformation one or many steps can be used.
    Those steps can be performed in memory only or their result
    can be serialized to the disk. Either in_memory or permanent or both steps
    should be provided in the init method.

    Parameters
    ----------
    transform_id: optional str
        Unique identifier of a transform.
    in_memory_steps: list of DfTransformStepConfig
        Those steps that are performed every time we read a dataframe from the disk.
    permanent_steps: list of DfTransformStepConfig
        Those steps result of which is persisted on the disk for future access.
    """
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

    def to_dict(self) -> Tuple[str, dict]:
        """
        Converts an instance to serializable dictionary

        Returns
        -------
        tuple of str and dict, transform_id and transform's dictionary representation.
        """
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
        """
        Builds DfTransformConfig instance out of serialized dictionary.

        Parameters
        ----------
        transform_id: optional str
            Unique identifier of a transform.
        transform_dict:
            Dictionary representation of DfTransformConfig

        Returns
        -------
        DfTransformConfig instance.
        """
        in_memory_transforms = DfTransformConfig._transform_step_configs_from(transform_dict=transform_dict,
                                                                              key=TRANSFORM_IN_MEMORY_KEY)
        permanent_transforms = DfTransformConfig._transform_step_configs_from(transform_dict=transform_dict,
                                                                              key=TRANSFORM_PERMANENT_KEY)

        result = DfTransformConfig(transform_id=transform_id,
                                   in_memory_steps=in_memory_transforms,
                                   permanent_steps=permanent_transforms)
        return result

    @staticmethod
    def _transform_step_configs_from(transform_dict: dict,
                                     key: str) -> Optional[List[DfTransformStepConfig]]:
        """
        When deserializing we need to transform steps into objects as well.

        Parameters
        ----------
        transform_dict:
            Dictionary representation of DfTransformConfig
        key: str
            Key for transform steps.

        Returns
        -------
        Optional list of DfTransformStepConfig objects.
        """
        transform_dicts = transform_dict.get(key)
        transforms = None
        if transform_dicts:
            transforms = [
                DfTransformStepConfig.from_dict(transform_dict=transform_dict)
                for transform_dict in transform_dicts
            ]

        return transforms
