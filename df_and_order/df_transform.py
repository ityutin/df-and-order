from typing import List, Optional, Tuple

from df_and_order.df_transform_step import DfTransformStepConfig


TRANSFORM_ID_KEY = 'transform_id'
TRANSFORM_DF_FORMAT_KEY = 'df_format'
TRANSFORM_SOURCE_ID_KEY = 'source_id'
TRANSFORM_SOURCE_IN_MEMORY_KEY = 'source_in_memory'
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
    transform_id: str
        Unique identifier of a transform.
    df_format: str
        What format to use to save permanent transforms
    source_id: optional str
        A transform may be based on another transform, source_id will be
        the desired transform_id you want to base on. If None - initial dataframe
        will be used by default.
    source_in_memory_steps: list of DfTransformStepConfig
        If source_id is not empty you may apply some transforms on a source_df first.
        It is useful when dealing with the initial dataframe.
    in_memory_steps: list of DfTransformStepConfig
        Those steps that are performed every time we read a dataframe from the disk.
    permanent_steps: list of DfTransformStepConfig
        Those steps result of which is persisted on the disk for future access.
    """
    def __init__(self,
                 transform_id: str,
                 df_format: Optional[str] = None,
                 source_id: Optional[str] = None,
                 source_in_memory_steps: Optional[List[DfTransformStepConfig]] = None,
                 in_memory_steps: Optional[List[DfTransformStepConfig]] = None,
                 permanent_steps: Optional[List[DfTransformStepConfig]] = None):
        assert in_memory_steps or permanent_steps, "Provide at least one type of transformations"

        self._transform_id = transform_id
        self._df_format = df_format
        self._source_id = source_id
        self._source_in_memory_steps = source_in_memory_steps or []
        self._in_memory_steps = in_memory_steps or []
        self._permanent_steps = permanent_steps or []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DfTransformConfig):
            return False

        result = self.transform_id == other.transform_id \
                 and self.df_format == other.df_format \
                 and self.source_id == other.source_id \
                 and self.source_in_memory_steps == other.source_in_memory_steps \
                 and self.in_memory_steps == other.in_memory_steps \
                 and self.permanent_steps == other.permanent_steps

        return result

    @property
    def transform_id(self) -> str:
        return self._transform_id

    @property
    def df_format(self) -> Optional[str]:
        return self._df_format

    @property
    def source_id(self) -> Optional[str]:
        return self._source_id

    @property
    def source_in_memory_steps(self) -> Optional[List[DfTransformStepConfig]]:
        return self._source_in_memory_steps

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

        if self._df_format:
            result[TRANSFORM_DF_FORMAT_KEY] = self._df_format

        if self._source_id:
            result[TRANSFORM_SOURCE_ID_KEY] = self._source_id

        if self._source_in_memory_steps:
            source_in_memory_dicts = [config.to_dict() for config in self._source_in_memory_steps]
            result[TRANSFORM_SOURCE_IN_MEMORY_KEY] = source_in_memory_dicts

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
        step_configs_getter = DfTransformConfig._transform_step_configs_from
        df_format = transform_dict.get(TRANSFORM_DF_FORMAT_KEY)
        source_id = transform_dict.get(TRANSFORM_SOURCE_ID_KEY)
        source_in_memory_transforms = step_configs_getter(transform_dict=transform_dict,
                                                          key=TRANSFORM_SOURCE_IN_MEMORY_KEY)
        in_memory_transforms = step_configs_getter(transform_dict=transform_dict,
                                                   key=TRANSFORM_IN_MEMORY_KEY)
        permanent_transforms = step_configs_getter(transform_dict=transform_dict,
                                                   key=TRANSFORM_PERMANENT_KEY)

        result = DfTransformConfig(transform_id=transform_id,
                                   df_format=df_format,
                                   source_id=source_id,
                                   source_in_memory_steps=source_in_memory_transforms,
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
        step_dicts = transform_dict.get(key)
        steps = None
        if step_dicts:
            steps = [
                DfTransformStepConfig.from_dict(step_dict=step_dict)
                for step_dict in step_dicts
            ]

        return steps
